# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""IR-extension kernels: ``@txl.kernel(passes=...)`` and the contract's extension check."""

import uuid

import pytest
import tvm_ffi
from tvm_ffi.dataclasses import py_class

import tirx_kernels.tirx_lite as txl
from tirx_kernels.tirx_lite import entry
from tirx_kernels.tirx_lite.low_level_ir import (
    LowLevelIRContractError,
    check_low_level_ir,
    inspect_low_level_ir,
)
from tvm_tirx import ir, tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.script.ir_builder.base import auto_emit_void_call
from tvm_tirx.script.ir_builder.stmt import region
from tvm_tirx.tirx.script import ir_builder as I

_emit_void = auto_emit_void_call(ir.Call)


class _Extension:
    """One namespace with a valued op, a statement op, a region op and a type."""

    def __init__(self):
        # Registrations are process-global; every test defines its own namespace.
        self.prefix = f"txl_test_{uuid.uuid4().hex[:8]}"

        @py_class(f"{self.prefix}.Box")
        class BoxType(ir.Type):
            dtype: ir.PrimType

        self.BoxType = BoxType
        pure = int(ir.CallEffectKind.Pure)
        self.scale = self._op("scale", ["value"], TFixedReturnType=ir.PrimType("float32"))
        self.box = self._op("box", ["value"], FInferType=lambda call: BoxType(call.args[0].ty))
        self.unbox = self._op("unbox", ["boxed"], FInferType=lambda call: call.args[0].ty.dtype)
        for op in (self.scale, self.box, self.unbox):
            op.set_attr("TCallEffectKind", pure)
        self.store = self._op("store", ["ptr", "value"], TFixedReturnType=ir.PrimType(""))
        self.store.set_attr("TCallEffectKind", int(ir.CallEffectKind.UpdateState))
        self.repeat = self._op(
            "repeat", ["n"], FRegionGetBodyParams=lambda call: [ir.Var("i", "int32")]
        )

    def _op(self, name, args, **attrs):
        full_name = f"{self.prefix}.{name}"
        tvm_ffi.get_global_func("ir.RegisterOp")(full_name, f"Test operator {name}.")
        op = ir.Op.get(full_name)
        op.set_signature(args=[(arg, f"The {arg} operand.") for arg in args])
        for key, value in attrs.items():
            op.set_attr(key, value)
        return op

    def kernel(self, **options):
        ext = self

        @txl.kernel(**options)
        def probe(
            x: txl.Tensor((txl.int64(),), "float32"), y: txl.Tensor((txl.int64(),), "float32")
        ):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
            lane = txl.lane_id()
            value = txl.local_scalar("float32")
            txl.ptx.ld.global_.f32(value, x.ptr_to([lane]))
            acc = txl.local_scalar("float32", init=txl.float32(0.0))
            with region(ext.repeat, [txl.int32(3)]):
                boxed = ir.Call(ext.box, [ir.Call(ext.scale, [value])])
                txl.assign(acc, acc + ir.Call(ext.unbox, [boxed]))
            _emit_void(ext.store, [y.ptr_to([lane]), acc])

        return probe

    def lower_exprs(self, func):
        """Rewrite the valued operators; leave statements alone."""

        def exprs(call):
            if call.op.same_as(self.scale):
                return call.args[0] * ir.FloatImm("float32", 2.0)
            if call.op.same_as(self.unbox):
                boxed = call.args[0]
                if isinstance(boxed, ir.Call) and boxed.op.same_as(self.box):
                    return boxed.args[0]
            return call

        return tvm_ffi.structural_map(func, [(ir.Call, exprs)])

    def lower_stmts(self, func, store=None):
        """Rewrite the region to a loop and the store with the ordinary builders."""

        def emit_store(ptr, value):
            with IRBuilder() as ib:
                with I.function_():
                    I.func_name_("lowered_store")
                    txl.ptx.st.global_.f32(ptr, value)
            return ib.get().body

        store = store or emit_store

        def stmts(stmt):
            if isinstance(stmt, ir.Evaluate) and isinstance(stmt.value, ir.Call):
                if stmt.value.op.same_as(self.store):
                    return store(*stmt.value.args)
            if isinstance(stmt, ir.RegionStmt) and stmt.op.same_as(self.repeat):
                (i,) = stmt.body_params
                zero, one = ir.IntImm("int32", 0), ir.IntImm("int32", 1)
                return ir.For(
                    i, zero, stmt.args[0], one, ir.ForKind.DEFAULT, stmt.body, loc=stmt.loc
                )
            return stmt

        return tvm_ffi.structural_map(func, [(ir.Stmt, stmts)])

    def lowering(self, func):
        return self.lower_stmts(self.lower_exprs(func))


def _extension_findings(report):
    return [finding for finding in report.violations if finding.kind.startswith("extension_")]


def test_unlowered_extension_constructs_are_contract_violations():
    ext = _Extension()
    with pytest.raises(LowLevelIRContractError) as error:
        ext.kernel()

    report = error.value.report
    found = {
        (finding.kind, finding.callee or finding.type_key)
        for finding in _extension_findings(report)
    }
    assert found == {
        ("extension_region", f"{ext.prefix}.repeat"),
        ("extension_call", f"{ext.prefix}.scale"),
        ("extension_call", f"{ext.prefix}.box"),
        ("extension_call", f"{ext.prefix}.unbox"),
        ("extension_call", f"{ext.prefix}.store"),
        ("extension_type", f"{ext.prefix}.Box"),
    }
    assert report.violations == tuple(_extension_findings(report))
    # A type that recurs on many values is reported once.
    assert [finding.kind for finding in report.violations].count("extension_type") == 1
    # Statement-level constructs point at the kernel source.
    spans = {finding.callee: finding.span for finding in report.violations}
    assert spans[f"{ext.prefix}.repeat"].startswith(__file__)
    assert spans[f"{ext.prefix}.store"].startswith(__file__)
    assert "passes=" in str(error.value)


def test_extension_type_on_a_parameter_is_reported():
    ext = _Extension()

    @txl.kernel(check_ir=False)
    def probe(boxed: ext.BoxType(ir.PrimType("float32"))):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))

    report = inspect_low_level_ir(probe.func)
    assert [(finding.kind, finding.type_key) for finding in report.violations] == [
        ("extension_type", f"{ext.prefix}.Box")
    ]


def test_passes_run_once_and_keep_the_traced_function():
    ext = _Extension()
    runs = []

    def counting(func):
        runs.append(func)
        return ext.lowering(func)

    kernel = ext.kernel(passes=[counting])

    assert len(runs) == 1 and runs[0].same_as(kernel.traced_func)
    assert not kernel.traced_func.same_as(kernel.func)
    assert _extension_findings(inspect_low_level_ir(kernel.traced_func))
    assert check_low_level_ir(kernel.func).ok
    assert kernel.mod["main"].same_as(kernel.func)
    assert kernel.func.attrs["global_symbol"] == "probe"
    assert len(runs) == 1


def test_kernel_without_passes_has_one_function():
    @txl.kernel()
    def probe(y: txl.Tensor((txl.int64(),), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
        txl.ptx.st.global_.f32(y.ptr_to([txl.lane_id()]), txl.float32(0.0))

    assert probe.traced_func is probe.func


def test_passes_run_with_the_kernel_session_active():
    ext = _Extension()
    seen = []

    def record(func):
        session = entry.current()
        seen.append((session, txl.lane_id()))
        return func

    kernel = ext.kernel(passes=[record, ext.lowering])

    ((session, lane),) = seen
    assert session is kernel.session
    assert lane.same_as(kernel.session.lane_id)
    assert entry.current(required=False) is None


def test_passes_accept_a_single_callable_and_tvm_passes():
    ext = _Extension()
    marker = tirx.transform.function_pass(
        lambda func, mod, ctx: func.with_attr("marker", 1), opt_level=0, name="marker"
    )

    assert ext.kernel(passes=[ext.lowering, marker]).func.attrs["marker"] == 1
    assert check_low_level_ir(ext.kernel(passes=ext.lowering).func).ok


def test_contract_checks_the_lowered_function():
    ext = _Extension()

    def to_tensor_store(ptr, value):
        # address_of(y[lane]) -> y[lane] = value: global memory through TensorStore.
        load = ptr.args[0]
        return ir.TensorStore(load.source, load.indices, value)

    def lowering(func):
        return ext.lower_stmts(ext.lower_exprs(func), store=to_tensor_store)

    with pytest.raises(LowLevelIRContractError) as error:
        ext.kernel(passes=[lowering])
    assert [(f.kind, f.scope) for f in error.value.report.violations] == [
        ("buffer_store", "global")
    ]

    # check_ir=False still lowers; it only skips the contract.
    kernel = ext.kernel(passes=[lowering], check_ir=False)
    assert not _extension_findings(inspect_low_level_ir(kernel.func))


def test_incomplete_lowering_reports_the_remaining_constructs():
    ext = _Extension()
    with pytest.raises(LowLevelIRContractError) as error:
        ext.kernel(passes=[ext.lower_exprs])

    assert {(f.kind, f.callee) for f in error.value.report.violations} == {
        ("extension_region", f"{ext.prefix}.repeat"),
        ("extension_call", f"{ext.prefix}.store"),
    }


def test_bad_passes_are_rejected_with_their_name():
    ext = _Extension()

    with pytest.raises(TypeError, match="must be callable"):
        txl.kernel(passes=[object()])

    def returns_none(func):
        return None

    with pytest.raises(TypeError, match="'returns_none' must return the rewritten"):
        ext.kernel(passes=[returns_none])

    def raises(func):
        raise ValueError("broken lowering")

    with pytest.raises(ValueError, match="broken lowering") as error:
        ext.kernel(passes=[raises])
    notes = "\n".join(getattr(error.value, "__notes__", []))
    if hasattr(error.value, "add_note"):
        assert "lowering pass 'raises'" in notes
    assert entry.current(required=False) is None
