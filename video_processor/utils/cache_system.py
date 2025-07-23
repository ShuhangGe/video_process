"""
Cache System - LRU cache and optimization for video processing performance.

This module provides comprehensive caching capabilities including memory caching,
disk caching, and cache invalidation strategies to optimize video processing performance.
"""

import os
import time
import hashlib
import pickle
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import lru_cache
from collections import OrderedDict
import threading
import json
import gzip

import torch


logger = logging.getLogger(__name__)


class MemoryCache:
    """
    Thread-safe LRU memory cache for video processing results.
    """
    
    def __init__(self, max_size: int = 1024):
        """Initialize memory cache with maximum size."""
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._access_count = 0
        self._hit_count = 0
        
        logger.debug(f"MemoryCache initialized with max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            self._access_count += 1
            
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._hit_count += 1
                logger.debug(f"Cache hit for key: {key[:50]}...")
                return value
            
            logger.debug(f"Cache miss for key: {key[:50]}...")
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self._cache:
                # Update existing key
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Remove least recently used item
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
                logger.debug(f"Evicted cache entry: {oldest_key[:50]}...")
            
            self._cache[key] = value
            logger.debug(f"Cached item with key: {key[:50]}...")
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                logger.debug(f"Removed cache entry: {key[:50]}...")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_count = 0
            self._hit_count = 0
            logger.info("Memory cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self._hit_count / max(self._access_count, 1)
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "access_count": self._access_count,
                "hit_count": self._hit_count,
                "hit_rate": hit_rate,
                "utilization": len(self._cache) / self.max_size,
            }


class DiskCache:
    """
    Disk-based cache with compression and TTL support.
    """
    
    def __init__(self, cache_dir: str, max_size: int = 10 * 1024 * 1024 * 1024, 
                 ttl: int = 7 * 24 * 3600, compression_level: int = 6):
        """Initialize disk cache."""
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.ttl = ttl
        self.compression_level = compression_level
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for metadata
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()
        
        logger.debug(f"DiskCache initialized at {self.cache_dir}")
    
    def _load_index(self):
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        if key not in self._index:
            return None
        
        entry = self._index[key]
        cache_path = self._get_cache_path(key)
        
        # Check TTL
        if time.time() - entry["created"] > self.ttl:
            self.remove(key)
            return None
        
        # Check if file exists
        if not cache_path.exists():
            self.remove(key)
            return None
        
        try:
            # Load and decompress
            with gzip.open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            entry["accessed"] = time.time()
            self._save_index()
            
            logger.debug(f"Disk cache hit for key: {key[:50]}...")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            self.remove(key)
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in disk cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            # Save compressed data
            with gzip.open(cache_path, 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(value, f)
            
            # Update index
            file_size = cache_path.stat().st_size
            current_time = time.time()
            
            self._index[key] = {
                "created": current_time,
                "accessed": current_time,
                "size": file_size,
                "path": str(cache_path),
            }
            
            # Check total cache size and cleanup if needed
            self._cleanup_if_needed()
            self._save_index()
            
            logger.debug(f"Cached to disk: {key[:50]}... ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
            return False
    
    def remove(self, key: str) -> bool:
        """Remove item from disk cache."""
        if key not in self._index:
            return False
        
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            
            del self._index[key]
            self._save_index()
            
            logger.debug(f"Removed from disk cache: {key[:50]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to remove from disk cache: {e}")
            return False
    
    def _cleanup_if_needed(self):
        """Cleanup cache if size exceeds limit."""
        total_size = sum(entry["size"] for entry in self._index.values())
        
        if total_size <= self.max_size:
            return
        
        # Sort by access time (least recently used first)
        sorted_keys = sorted(
            self._index.keys(),
            key=lambda k: self._index[k]["accessed"]
        )
        
        # Remove oldest entries until under limit
        for key in sorted_keys:
            self.remove(key)
            total_size = sum(entry["size"] for entry in self._index.values())
            if total_size <= self.max_size * 0.8:  # Leave some headroom
                break
        
        logger.info(f"Disk cache cleanup completed, size: {total_size / (1024*1024):.1f}MB")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self._index.keys()):
            self.remove(key)
        logger.info("Disk cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry["size"] for entry in self._index.values())
        return {
            "entries": len(self._index),
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size / (1024 * 1024),
            "utilization": total_size / self.max_size,
            "cache_dir": str(self.cache_dir),
        }


class CacheSystem:
    """
    Unified cache system combining memory and disk caching.
    """
    
    def __init__(self, config):
        """Initialize cache system with configuration."""
        self.config = config
        self.cache_config = config.cache
        
        # Initialize caches if enabled
        if self.cache_config.enable_memory_cache:
            self.memory_cache = MemoryCache(self.cache_config.memory_cache_size)
        else:
            self.memory_cache = None
        
        if self.cache_config.enable_disk_cache:
            self.disk_cache = DiskCache(
                cache_dir=self.cache_config.cache_dir,
                max_size=self.cache_config.max_cache_size,
                ttl=self.cache_config.cache_ttl,
                compression_level=self.cache_config.compression_level
            )
        else:
            self.disk_cache = None
        
        # Async write queue for disk cache
        if self.cache_config.async_cache_writes and self.disk_cache:
            self._write_queue = asyncio.Queue()
            self._writer_task = None
        else:
            self._write_queue = None
        
        logger.info(
            f"CacheSystem initialized: memory={self.memory_cache is not None}, "
            f"disk={self.disk_cache is not None}, async={self._write_queue is not None}"
        )
    
    def _generate_cache_key(self, video_config: Dict[str, Any], 
                          processing_params: Dict[str, Any] = None) -> str:
        """Generate cache key from video configuration and processing parameters."""
        # Combine relevant configuration parameters
        key_data = {
            "video_path": video_config.get("video"),
            "video_start": video_config.get("video_start"),
            "video_end": video_config.get("video_end"),
            "nframes": video_config.get("nframes"),
            "fps": video_config.get("fps"),
            "resized_height": video_config.get("resized_height"),
            "resized_width": video_config.get("resized_width"),
        }
        
        if processing_params:
            key_data["processing"] = processing_params
        
        # Create hash from sorted key data
        key_str = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()
        
        return cache_key
    
    def get(self, video_config: Dict[str, Any], 
           processing_params: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached result."""
        if not self._is_cache_enabled():
            return None
        
        cache_key = self._generate_cache_key(video_config, processing_params)
        
        # Try memory cache first
        if self.memory_cache:
            result = self.memory_cache.get(cache_key)
            if result is not None:
                return result
        
        # Try disk cache
        if self.disk_cache:
            result = self.disk_cache.get(cache_key)
            if result is not None:
                # Also cache in memory for faster future access
                if self.memory_cache:
                    self.memory_cache.put(cache_key, result)
                return result
        
        return None
    
    def put(self, video_config: Dict[str, Any], result: Any,
           processing_params: Dict[str, Any] = None) -> None:
        """Put result in cache."""
        if not self._is_cache_enabled():
            return
        
        cache_key = self._generate_cache_key(video_config, processing_params)
        
        # Cache in memory
        if self.memory_cache:
            self.memory_cache.put(cache_key, result)
        
        # Cache to disk (async if enabled)
        if self.disk_cache:
            if self._write_queue is not None:
                # Async write
                asyncio.create_task(self._async_disk_write(cache_key, result))
            else:
                # Sync write
                self.disk_cache.put(cache_key, result)
    
    async def _async_disk_write(self, cache_key: str, result: Any) -> None:
        """Async disk cache write."""
        try:
            await self._write_queue.put((cache_key, result))
            if self._writer_task is None:
                self._writer_task = asyncio.create_task(self._disk_writer_worker())
        except Exception as e:
            logger.warning(f"Failed to queue async cache write: {e}")
    
    async def _disk_writer_worker(self) -> None:
        """Worker task for async disk writes."""
        while True:
            try:
                cache_key, result = await self._write_queue.get()
                self.disk_cache.put(cache_key, result)
                self._write_queue.task_done()
            except Exception as e:
                logger.warning(f"Async cache write failed: {e}")
    
    def invalidate(self, video_config: Dict[str, Any], 
                  processing_params: Dict[str, Any] = None) -> None:
        """Invalidate cached result."""
        cache_key = self._generate_cache_key(video_config, processing_params)
        
        if self.memory_cache:
            self.memory_cache.remove(cache_key)
        
        if self.disk_cache:
            self.disk_cache.remove(cache_key)
        
        logger.debug(f"Invalidated cache for key: {cache_key[:20]}...")
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.disk_cache:
            self.disk_cache.clear()
        
        logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "enabled": self._is_cache_enabled(),
            "memory_cache": None,
            "disk_cache": None,
        }
        
        if self.memory_cache:
            stats["memory_cache"] = self.memory_cache.get_stats()
        
        if self.disk_cache:
            stats["disk_cache"] = self.disk_cache.get_stats()
        
        return stats
    
    def _is_cache_enabled(self) -> bool:
        """Check if any caching is enabled."""
        return (self.cache_config.enable_cache and 
                (self.memory_cache is not None or self.disk_cache is not None))
    
    def warmup_cache(self, video_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Warmup cache with common video configurations.
        
        Args:
            video_configs: List of video configurations to pre-process
            
        Returns:
            Warmup statistics
        """
        warmup_stats = {
            "total_configs": len(video_configs),
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }
        
        for video_config in video_configs:
            try:
                result = self.get(video_config)
                if result is not None:
                    warmup_stats["cache_hits"] += 1
                else:
                    warmup_stats["cache_misses"] += 1
            except Exception as e:
                logger.warning(f"Cache warmup error for config {video_config}: {e}")
                warmup_stats["errors"] += 1
        
        logger.info(
            f"Cache warmup completed: {warmup_stats['cache_hits']} hits, "
            f"{warmup_stats['cache_misses']} misses, {warmup_stats['errors']} errors"
        )
        
        return warmup_stats
    
    def optimize_cache(self) -> Dict[str, Any]:
        """
        Optimize cache performance and cleanup unused entries.
        
        Returns:
            Optimization statistics
        """
        optimization_stats = {
            "memory_before": 0,
            "memory_after": 0,
            "disk_before": 0,
            "disk_after": 0,
        }
        
        # Get initial stats
        if self.memory_cache:
            optimization_stats["memory_before"] = self.memory_cache.get_stats()["size"]
        
        if self.disk_cache:
            disk_stats = self.disk_cache.get_stats()
            optimization_stats["disk_before"] = disk_stats["entries"]
            
            # Trigger disk cache cleanup
            self.disk_cache._cleanup_if_needed()
        
        # Get final stats
        if self.memory_cache:
            optimization_stats["memory_after"] = self.memory_cache.get_stats()["size"]
        
        if self.disk_cache:
            disk_stats = self.disk_cache.get_stats()
            optimization_stats["disk_after"] = disk_stats["entries"]
        
        logger.info(f"Cache optimization completed: {optimization_stats}")
        return optimization_stats


def create_cache_decorator(cache_system: CacheSystem):
    """
    Create a decorator for caching function results.
    
    Args:
        cache_system: Cache system instance
        
    Returns:
        Decorator function
    """
    def cache_decorator(processing_params: Dict[str, Any] = None):
        def decorator(func):
            def wrapper(video_config: Dict[str, Any], *args, **kwargs):
                # Try to get from cache
                cached_result = cache_system.get(video_config, processing_params)
                if cached_result is not None:
                    return cached_result
                
                # Compute result and cache it
                result = func(video_config, *args, **kwargs)
                cache_system.put(video_config, result, processing_params)
                
                return result
            return wrapper
        return decorator
    return cache_decorator 