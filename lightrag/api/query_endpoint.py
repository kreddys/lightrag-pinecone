# lightrag/api/query_endpoint.py

from fastapi import HTTPException
import sentry_sdk
from lightrag import QueryParam
import logging

class QueryEndpoint:
    def __init__(self, rag):
        self.rag = rag

    async def query_text(self, request):
        """
        Handle a POST request to process user queries using RAG capabilities.
        """

        logging.debug(f"Processing query request: {request}")
        logging.debug(f"Query text: {request.query}")
        logging.debug(f"Query mode: {request.mode}")
        logging.debug(f"Stream enabled: {request.stream}")
        logging.debug(f"Only need context: {request.only_need_context}")
        
        # Start a new transaction for this query
        with sentry_sdk.start_transaction(
            op="query",
            name="rag_query",
            description=f"RAG Query: {request.query[:50]}..."
        ) as transaction:
            try:
                # Add request context to Sentry
                sentry_sdk.set_context("query_request", {
                    "query": request.query,
                    "mode": request.mode,
                    "stream": request.stream,
                    "only_need_context": request.only_need_context
                })

                # Add breadcrumb for query start
                sentry_sdk.add_breadcrumb(
                    category="query",
                    message="Starting query processing",
                    level="info",
                    data={
                        "query": request.query,
                        "mode": str(request.mode),
                    }
                )

                # Process RAG query with its own span
                with transaction.start_child(
                    op="rag_processing",
                    description="Processing RAG query"
                ) as rag_span:
                    response = await self.rag.aquery(
                        request.query,
                        param=QueryParam(
                            mode=request.mode,
                            stream=request.stream,
                            only_need_context=request.only_need_context,
                        ),
                    )

                    # Add breadcrumb for RAG completion
                    sentry_sdk.add_breadcrumb(
                        category="query",
                        message="RAG processing completed",
                        level="info"
                    )

                # Handle different response types
                if isinstance(response, str):
                    # Cache hit case
                    sentry_sdk.add_breadcrumb(
                        category="query",
                        message="Cache hit - direct response",
                        level="info"
                    )
                    transaction.set_tag("response_type", "cache_hit")
                    return {"response": response}

                # Handle streaming response
                with transaction.start_child(
                    op="response_processing",
                    description="Processing streaming response"
                ) as stream_span:
                    result = ""
                    chunk_count = 0

                    if request.stream:
                        stream_span.set_tag("streaming", "true")
                        async for chunk in response:
                            result += chunk
                            chunk_count += 1
                            
                            # Add breadcrumb every N chunks
                            if chunk_count % 10 == 0:
                                sentry_sdk.add_breadcrumb(
                                    category="streaming",
                                    message=f"Processed {chunk_count} chunks",
                                    level="info"
                                )
                    else:
                        stream_span.set_tag("streaming", "false")
                        async for chunk in response:
                            result += chunk
                            chunk_count += 1

                    # Set response metrics
                    stream_span.set_data("total_chunks", chunk_count)
                    stream_span.set_data("response_length", len(result))

                    sentry_sdk.add_breadcrumb(
                        category="query",
                        message="Response generation completed",
                        level="info",
                        data={
                            "chunks_processed": chunk_count,
                            "response_length": len(result)
                        }
                    )

                    return {"response": result}

            except Exception as e:
                # Add error context
                sentry_sdk.set_context("error_details", {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })

                # Add error breadcrumb
                sentry_sdk.add_breadcrumb(
                    category="error",
                    message="Query processing failed",
                    level="error",
                    data={
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )

                # Mark transaction as failed
                transaction.set_status("internal_error")
                
                # Capture the exception with full context
                sentry_sdk.capture_exception(e)
                
                # Raise HTTP exception
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )

            finally:
                # Add final metrics to transaction
                transaction.set_data("query_length", len(request.query))
                transaction.set_tag("query_mode", str(request.mode))
                transaction.set_tag("streaming_enabled", str(request.stream))