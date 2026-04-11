from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class CloudScalerClient:
    """Sync client for CloudScalerEnv OpenEnv API."""

    def __init__(self, base_url: str = "http://127.0.0.1:7860", timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def health(self) -> Dict[str, Any]:
        response = self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        services: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        if max_steps is not None:
            payload["max_steps"] = max_steps
        if services is not None:
            payload["services"] = services

        response = self._client.post(f"{self.base_url}/reset", json=payload if payload else None)
        response.raise_for_status()
        return response.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.post(f"{self.base_url}/step", json=action)
        response.raise_for_status()
        return response.json()

    def state(self) -> Dict[str, Any]:
        response = self._client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()


class AsyncCloudScalerClient:
    """Async client for CloudScalerEnv OpenEnv API."""

    def __init__(self, base_url: str = "http://127.0.0.1:7860", timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> Dict[str, Any]:
        response = await self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        services: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        if max_steps is not None:
            payload["max_steps"] = max_steps
        if services is not None:
            payload["services"] = services

        response = await self._client.post(
            f"{self.base_url}/reset", json=payload if payload else None
        )
        response.raise_for_status()
        return response.json()

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post(f"{self.base_url}/step", json=action)
        response.raise_for_status()
        return response.json()

    async def state(self) -> Dict[str, Any]:
        response = await self._client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()
