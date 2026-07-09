# mypy: allow-untyped-defs
import hashlib
import logging
from collections.abc import Sequence
from typing import cast

from torch._inductor.utils import Placeholder
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import ComputedBuffer, FlyDSLTemplateBuffer, Pointwise
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...select_algorithm import PartialRender
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)


class FlyDSLScheduling(BaseScheduling):
    """Scheduling implementation for FlyDSL template kernels."""

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_flydsl_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, FlyDSLTemplateBuffer
        )

    def is_flydsl_fused_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, FusedSchedulerNode) and self.is_flydsl_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if not config.epilogue_fusion:
            return False
        if not self.is_flydsl_template(node1):
            return False
        if node2.has_aliasing_or_mutation() or node2.is_reduction():
            return False

        template_node = cast(SchedulerNode, node1)
        ir_node = template_node.node
        if not isinstance(ir_node, FlyDSLTemplateBuffer):
            return False

        reads = OrderedSet()
        for scheduler_node in node2.get_nodes():
            node = scheduler_node.node
            if not isinstance(node, ComputedBuffer) or not isinstance(
                node.data, Pointwise
            ):
                return False
            if not V.graph.sizevars.statically_known_list_equals(
                node.get_size(), ir_node.get_size()
            ):
                return False
            reads |= OrderedSet(rd.name for rd in scheduler_node.read_writes.reads)

        # Initial hgemm epilogue ABI supports accumulator-only expressions.
        # Extra tensor reads (bias/mask/etc.) need launch arguments first.
        return reads == OrderedSet([ir_node.get_name()])

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def define_kernel(self, src_code_str: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code

        if src_code_str in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code_str]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )

            kernel_hash = hashlib.sha256(src_code_str.encode("utf-8")).hexdigest()[:8]
            if fused_name == "fused":
                kernel_name = f"flydsl_{kernel_hash}"
            else:
                kernel_name = f"flydsl_{fused_name}_{kernel_hash}"
            wrapper.src_to_kernel[src_code_str] = kernel_name
            src_code_str = src_code_str.replace(
                str(Placeholder.KERNEL_NAME), kernel_name
            )

            _, _, kernel_path = get_path(code_hash(src_code_str), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.flydsl({kernel_name!r}, r'''")
            compile_wrapper.splice(src_code_str, strip=True)
            compile_wrapper.writeline("''')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        if not self.is_flydsl_template(template_node):
            raise AssertionError(
                "Template node passed to FlyDSLScheduling.codegen_template must be a "
                "SchedulerNode that wraps a FlyDSLTemplateBuffer"
            )
        if prologue_nodes:
            raise AssertionError("FlyDSL doesn't support prologue fusion yet")

        template_node = cast(SchedulerNode, template_node)
        ftb: FlyDSLTemplateBuffer = cast(FlyDSLTemplateBuffer, template_node.node)

        output_node = epilogue_nodes[-1].node if epilogue_nodes else ftb
        kernel, render = ftb.make_kernel_render(output_node)  # type: ignore[misc]
        kernel.original_output_name = ftb.get_name()
        kernel.epilogue_nodes = epilogue_nodes
        template_node.mark_run()
        src_code = render()
        if isinstance(src_code, PartialRender):
            src_code_str = src_code.finalize_all()
        else:
            src_code_str = src_code

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code_str, node_schedule)
        self.codegen_comment(node_schedule, kernel_name)
        if epilogue_nodes:
            V.graph.removed_buffers.add(ftb.get_name())
            for node in epilogue_nodes[:-1]:
                V.graph.removed_buffers.add(node.get_name())
            for node in epilogue_nodes:
                node.mark_run()
        kernel.call_kernel(kernel_name, output_node)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()
