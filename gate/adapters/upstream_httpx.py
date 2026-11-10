"""httpx transport, with send-progress evidence for ``not_sent``.

No hidden retries and no redirects: every upstream execution is one attempt.
Only failures during connection establishment (or waiting for a pooled
connection) count as proof that no request bytes were sent. A write error, a
read timeout or a failure on a reused connection is not proof.
"""

from collections.abc import AsyncIterator, Mapping, Sequence

import httpx

from gate.app.ports import FailureKind, UpstreamFailure, UpstreamReadError


class HttpxResponse:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status = response.status_code
        self.headers: Sequence[tuple[str, str]] = [
            (k.decode("latin-1"), v.decode("latin-1")) for k, v in response.headers.raw
        ]

    async def iter_raw(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.aiter_raw():
                if chunk:
                    yield chunk
        except httpx.TimeoutException as exc:
            raise UpstreamReadError(FailureKind.TIMEOUT) from exc
        except (httpx.TransportError, httpx.StreamError) as exc:
            raise UpstreamReadError(FailureKind.CONNECTION) from exc

    async def aclose(self) -> None:
        await self._response.aclose()


class HttpxUpstream:
    def __init__(self, *, connect_timeout_s: float, read_timeout_s: float) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout_s,
            read=read_timeout_s,
            write=read_timeout_s,
            pool=connect_timeout_s,
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=httpx.AsyncHTTPTransport(retries=0),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=50),
        )

    async def send(self, url: str, headers: Mapping[str, str], body: bytes) -> HttpxResponse:
        request = self._client.build_request("POST", url, headers=dict(headers), content=body)
        try:
            response = await self._client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            kind = (
                FailureKind.TIMEOUT
                if isinstance(exc, httpx.TimeoutException)
                else FailureKind.CONNECT
            )
            raise UpstreamFailure(kind, not_sent=True) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamFailure(FailureKind.TIMEOUT, not_sent=False) from exc
        except httpx.TransportError as exc:
            raise UpstreamFailure(FailureKind.CONNECTION, not_sent=False) from exc
        return HttpxResponse(response)

    async def aclose(self) -> None:
        await self._client.aclose()
