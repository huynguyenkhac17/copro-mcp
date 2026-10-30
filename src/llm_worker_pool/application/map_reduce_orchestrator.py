"""Orchestrates Map-Reduce execution for large contexts."""

from __future__ import annotations

import asyncio

from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.application.dto import DelegateTaskRequest, DelegateTaskResult
from llm_worker_pool.domain.chunker import Chunker
from llm_worker_pool.domain.file_context import FileContextResolver, resolve_request_context
from llm_worker_pool.domain.token_estimate import CONTEXT_RESERVE_RATIO
from llm_worker_pool.domain.validators import citation_warnings


class MapReduceOrchestrator:
    def __init__(
        self,
        delegate_task_use_case: DelegateTaskUseCase,
        chunker: Chunker,
        file_context_resolver: FileContextResolver,
    ) -> None:
        self._use_case = delegate_task_use_case
        self._chunker = chunker
        self._file_context_resolver = file_context_resolver

    async def execute(self, request: DelegateTaskRequest) -> DelegateTaskResult:
        if not request.context and not request.file_paths and not request.urls:
            # Nothing to map-reduce over, just call directly
            return await self._use_case.execute(request)

        # 1. Resolve context first. The task profile is applied here too, so chunk
        # sizing uses the budget the profile picks.
        request, _ = self._use_case.apply_task_profile(request)
        profile = self._use_case.resolve_budget(request.budget)

        combined_context, file_warnings = await resolve_request_context(
            self._file_context_resolver,
            context=request.context,
            file_paths=request.file_paths,
            query=request.task,
            urls=request.urls,
            url_fetcher=self._use_case.url_fetcher,
        )

        if not combined_context:
            return await self._use_case.execute(request)

        max_context_tokens = int(profile.options.num_ctx * (1 - CONTEXT_RESERVE_RATIO))
        
        chunks = self._chunker.split(
            combined_context, 
            chunk_size_tokens=max_context_tokens, 
            overlap_tokens=max_context_tokens // 10
        )
        
        if len(chunks) == 1:
            # Fits in one chunk, use standard execution
            return await self._use_case.execute(request)
            
        # 2. Map Phase
        map_tasks = []
        for i, chunk in enumerate(chunks):
            map_req = DelegateTaskRequest(
                task=f"Extract all information relevant to the following task:\n{request.task}",
                provider=request.provider,
                budget=request.budget,
                context=f"Part {i+1} of {len(chunks)}:\n{chunk}",
                output_format="text", 
                task_type=request.task_type,
            )
            map_tasks.append(self._use_case.execute(map_req))
            
        map_results = await asyncio.gather(*map_tasks, return_exceptions=True)
        
        valid_map_results = []
        for i, res in enumerate(map_results):
            if isinstance(res, Exception):
                file_warnings.append(f"map_reduce_failed_part:{i}")
            elif not res.success:
                file_warnings.append(f"map_reduce_failed_part:{i}:{res.error}")
            else:
                valid_map_results.append(f"--- Extracted from Part {i+1} ---\n{res.raw_text}")
                
        if not valid_map_results:
            return DelegateTaskResult(
                success=False,
                error="All map tasks failed during map-reduce.",
                warnings=file_warnings,
            )
            
        # 3. Reduce Phase
        reduce_context = "\n\n".join(valid_map_results)
        reduce_req = DelegateTaskRequest(
            task=f"Synthesize the provided information into a final answer for the task:\n{request.task}",
            provider=request.provider,
            budget=request.budget,
            context=reduce_context,
            output_format=request.output_format,
            json_schema=request.json_schema,
            task_type=request.task_type,
        )
        
        final_result = await self._use_case.execute(reduce_req)

        if final_result.success and final_result.verification is not None:
            # The reduce call only saw the map extracts, which carry no file headers -
            # re-check the citations against the files they actually point into.
            final_result.warnings = [
                w for w in final_result.warnings if not w.startswith("citation_unverified:")
            ]
            final_result.verification = self._use_case.verify_citations(
                final_result.raw_text, final_result.content, combined_context
            )
            final_result.warnings.extend(citation_warnings(final_result.verification))
        
        # Add map-reduce warnings
        final_result.warnings.extend(file_warnings)
        final_result.warnings.append(f"map_reduce_chunks:{len(chunks)}")
        
        return final_result
