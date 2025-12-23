"""
OpenAQ API v3 Client Wrapper

Provides async and sync interfaces for fetching air quality data from OpenAQ.
Handles rate limiting, caching, and error recovery.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import hashlib

import httpx
import pandas as pd
from loguru import logger


@dataclass
class OpenAQConfig:
    """Configuration for OpenAQ API client."""
    api_key: str
    base_url: str = "https://api.openaq.org/v3"
    cache_dir: Optional[Path] = None
    cache_ttl: int = 3600  # seconds
    rate_limit_calls: int = 100
    rate_limit_period: int = 60  # seconds
    timeout: int = 30


class OpenAQClient:
    """
    Async client for OpenAQ API v3.
    
    Features:
    - Automatic pagination handling
    - Response caching
    - Rate limiting
    - Retry logic with exponential backoff
    
    Example:
        >>> async with OpenAQClient(api_key="your-key") as client:
        ...     locations = await client.get_locations(city="Delhi")
        ...     measurements = await client.get_measurements(
        ...         location_id=12345,
        ...         date_from="2024-01-01",
        ...         date_to="2024-01-31"
        ...     )
    """
    
    PARAMETER_MAP = {
        "pm25": 2,
        "pm10": 1,
        "o3": 3,
        "co": 4,
        "no2": 5,
        "so2": 6,
        "bc": 7,  # Black carbon
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[OpenAQConfig] = None
    ):
        """Initialize the OpenAQ client."""
        if config:
            self.config = config
        else:
            self.config = OpenAQConfig(
                api_key=api_key or os.environ.get("OPENAQ_API_KEY", ""),
                cache_dir=Path("data/cache") if os.path.exists("data") else None
            )
        
        if not self.config.api_key:
            logger.warning(
                "No OpenAQ API key provided. Set OPENAQ_API_KEY environment variable "
                "or pass api_key parameter. Rate limits will be restricted."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = asyncio.Semaphore(self.config.rate_limit_calls)
        
        # Setup cache
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key
            
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout
            )
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from request parameters."""
        param_str = json.dumps(params, sort_keys=True)
        hash_input = f"{endpoint}:{param_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached response if valid."""
        if not self.config.cache_dir:
            return None
        
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        # Check TTL
        mtime = cache_file.stat().st_mtime
        if datetime.now().timestamp() - mtime > self.config.cache_ttl:
            return None
        
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    
    def _cache_response(self, cache_key: str, response: Dict):
        """Cache API response."""
        if not self.config.cache_dir:
            return
        
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(response, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """Make API request with rate limiting, caching, and retries."""
        await self._ensure_client()
        params = params or {}
        
        # Check cache
        if use_cache and method.upper() == "GET":
            cache_key = self._get_cache_key(endpoint, params)
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.debug(f"Cache hit for {endpoint}")
                return cached
        
        # Rate limiting
        async with self._rate_limiter:
            for attempt in range(3):
                try:
                    response = await self._client.request(
                        method=method,
                        url=endpoint,
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Cache successful response
                    if use_cache and method.upper() == "GET":
                        self._cache_response(cache_key, data)
                    
                    return data
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    elif e.response.status_code >= 500:
                        # Server error - retry
                        await asyncio.sleep(1)
                    else:
                        raise
                except httpx.TimeoutException:
                    logger.warning(f"Request timeout, attempt {attempt + 1}/3")
                    await asyncio.sleep(1)
        
        raise Exception(f"Failed to fetch {endpoint} after 3 attempts")
    
    async def _paginate(
        self,
        endpoint: str,
        params: Dict,
        limit: int = 1000
    ) -> List[Dict]:
        """Handle paginated API responses."""
        all_results = []
        page = 1
        params["limit"] = min(limit, 1000)
        
        while True:
            params["page"] = page
            response = await self._request("GET", endpoint, params)
            
            results = response.get("results", [])
            all_results.extend(results)
            
            # Check if more pages
            meta = response.get("meta", {})
            total_found = meta.get("found", len(results))
            
            if len(all_results) >= total_found or len(results) < params["limit"]:
                break
            
            page += 1
            if page > 100:  # Safety limit
                logger.warning("Reached pagination limit")
                break
        
        return all_results
    
    # =========================================================================
    # API Endpoints
    # =========================================================================
    
    async def get_locations(
        self,
        city: Optional[str] = None,
        country: Optional[str] = None,
        coordinates: Optional[tuple] = None,
        radius: int = 25000,
        parameters: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get monitoring locations.
        
        Args:
            city: Filter by city name
            country: Filter by country code (e.g., "IN", "US")
            coordinates: (latitude, longitude) for spatial query
            radius: Radius in meters for coordinate search
            parameters: Filter by parameters (e.g., ["pm25", "no2"])
            limit: Maximum results to return
        
        Returns:
            List of location dictionaries
        """
        params = {}
        
        if city:
            params["city"] = city
        if country:
            params["country"] = country
        if coordinates:
            lat, lon = coordinates
            params["coordinates"] = f"{lon},{lat}"
            params["radius"] = radius
        if parameters:
            # Convert parameter names to IDs
            param_ids = [self.PARAMETER_MAP.get(p, p) for p in parameters]
            params["parameters_id"] = ",".join(map(str, param_ids))
        
        return await self._paginate("/locations", params, limit)
    
    async def get_sensors(
        self,
        location_id: int
    ) -> List[Dict]:
        """Get sensors for a specific location."""
        response = await self._request(
            "GET",
            f"/locations/{location_id}/sensors"
        )
        return response.get("results", [])
    
    async def get_measurements(
        self,
        sensors_id: Optional[int] = None,
        location_id: Optional[int] = None,
        parameter: Optional[str] = None,
        date_from: Optional[Union[str, datetime]] = None,
        date_to: Optional[Union[str, datetime]] = None,
        data: str = "hours",  # hours, days, years
        limit: int = 10000
    ) -> List[Dict]:
        """
        Get measurement data.
        
        Args:
            sensors_id: Specific sensor ID
            location_id: Location ID (will fetch all sensors)
            parameter: Parameter name (e.g., "pm25")
            date_from: Start date (ISO format or datetime)
            date_to: End date (ISO format or datetime)
            data: Temporal resolution ("hours", "days", "years")
            limit: Maximum results
        
        Returns:
            List of measurement dictionaries
        """
        if sensors_id:
            endpoint = f"/sensors/{sensors_id}/{data}"
        elif location_id:
            endpoint = f"/locations/{location_id}/measurements"
        else:
            raise ValueError("Must provide sensors_id or location_id")
        
        params = {"limit": min(limit, 1000)}
        
        if date_from:
            if isinstance(date_from, datetime):
                date_from = date_from.isoformat()
            params["date_from"] = date_from
        
        if date_to:
            if isinstance(date_to, datetime):
                date_to = date_to.isoformat()
            params["date_to"] = date_to
        
        if parameter:
            params["parameters_id"] = self.PARAMETER_MAP.get(parameter, parameter)
        
        return await self._paginate(endpoint, params, limit)
    
    async def get_latest(
        self,
        parameter: Optional[str] = None,
        city: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get latest measurements for all locations."""
        params = {}
        
        if parameter:
            param_id = self.PARAMETER_MAP.get(parameter, parameter)
            endpoint = f"/parameters/{param_id}/latest"
        else:
            endpoint = "/latest"
        
        if city:
            params["city"] = city
        
        return await self._paginate(endpoint, params, limit)
    
    async def get_countries(self) -> List[Dict]:
        """Get list of available countries."""
        return await self._paginate("/countries", {})
    
    async def get_parameters(self) -> List[Dict]:
        """Get list of available parameters."""
        response = await self._request("GET", "/parameters")
        return response.get("results", [])


class OpenAQDataFetcher:
    """
    High-level data fetcher for building ML datasets.
    
    Fetches, cleans, and structures data from multiple locations
    into pandas DataFrames suitable for time series modeling.
    
    Example:
        >>> fetcher = OpenAQDataFetcher(api_key="your-key")
        >>> df = await fetcher.fetch_city_data(
        ...     city="Delhi",
        ...     parameters=["pm25", "no2"],
        ...     days=30
        ... )
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize data fetcher."""
        self.client = OpenAQClient(api_key=api_key)
    
    async def fetch_city_data(
        self,
        city: str,
        parameters: List[str] = ["pm25", "pm10", "no2", "o3"],
        days: int = 30,
        min_coverage: float = 0.8
    ) -> pd.DataFrame:
        """
        Fetch and structure data for a city.
        
        Args:
            city: City name
            parameters: List of parameters to fetch
            days: Number of days of historical data
            min_coverage: Minimum data completeness threshold
        
        Returns:
            DataFrame with columns: datetime, location_id, lat, lon, 
                                   pm25, pm10, no2, o3, etc.
        """
        async with self.client:
            # Get locations in the city
            locations = await self.client.get_locations(
                city=city,
                parameters=parameters
            )
            
            if not locations:
                logger.warning(f"No locations found for {city}")
                return pd.DataFrame()
            
            logger.info(f"Found {len(locations)} locations in {city}")
            
            # Date range
            date_to = datetime.utcnow()
            date_from = date_to - timedelta(days=days)
            
            # Fetch measurements for each location
            all_data = []
            
            for loc in locations:
                loc_id = loc.get("id")
                coords = loc.get("coordinates", {})
                
                try:
                    measurements = await self.client.get_measurements(
                        location_id=loc_id,
                        date_from=date_from,
                        date_to=date_to,
                        data="hours"
                    )
                    
                    for m in measurements:
                        period = m.get("period", {})
                        all_data.append({
                            "datetime": period.get("datetimeFrom", {}).get("utc"),
                            "location_id": loc_id,
                            "location_name": loc.get("name"),
                            "latitude": coords.get("latitude"),
                            "longitude": coords.get("longitude"),
                            "parameter": m.get("parameter", {}).get("name"),
                            "value": m.get("value"),
                            "unit": m.get("parameter", {}).get("units")
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch data for location {loc_id}: {e}")
            
            if not all_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            # Pivot parameters to columns
            df_pivot = df.pivot_table(
                index=["datetime", "location_id", "location_name", "latitude", "longitude"],
                columns="parameter",
                values="value",
                aggfunc="mean"
            ).reset_index()
            
            # Flatten column names
            df_pivot.columns = [
                col if isinstance(col, str) else col
                for col in df_pivot.columns
            ]
            
            # Sort by datetime
            df_pivot = df_pivot.sort_values("datetime")
            
            # Calculate coverage and filter
            expected_hours = days * 24
            coverage = df_pivot.groupby("location_id").size() / expected_hours
            valid_locations = coverage[coverage >= min_coverage].index
            
            df_filtered = df_pivot[df_pivot["location_id"].isin(valid_locations)]
            
            logger.info(
                f"Fetched {len(df_filtered)} records from "
                f"{len(valid_locations)} locations with ≥{min_coverage*100:.0f}% coverage"
            )
            
            return df_filtered
    
    async def fetch_multi_city_data(
        self,
        cities: List[str],
        parameters: List[str] = ["pm25", "pm10", "no2", "o3"],
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple cities."""
        results = {}
        
        for city in cities:
            logger.info(f"Fetching data for {city}...")
            df = await self.fetch_city_data(city, parameters, days)
            results[city] = df
        
        return results


# Synchronous wrapper for non-async contexts
def fetch_city_data_sync(
    city: str,
    api_key: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Synchronous wrapper for fetch_city_data."""
    fetcher = OpenAQDataFetcher(api_key=api_key)
    return asyncio.run(fetcher.fetch_city_data(city, **kwargs))


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        fetcher = OpenAQDataFetcher()
        
        # Fetch Delhi data
        df = await fetcher.fetch_city_data(
            city="Delhi",
            parameters=["pm25", "pm10", "no2"],
            days=7
        )
        
        print(f"Shape: {df.shape}")
        print(df.head())
    
    asyncio.run(main())
