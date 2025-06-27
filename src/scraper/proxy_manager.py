import logging
import random
import time
from typing import Optional, List, Dict

from ..core.config import AppConfig

logger = logging.getLogger(__name__)

class ProxyManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.proxies: List[str] = self.config.proxy_list
        # Health tracking: {'proxy_url': {'status': 'healthy'/'unhealthy', 'last_fail_time': timestamp}}
        self.proxy_health: Dict[str, Dict] = {
            proxy: {'status': 'healthy', 'last_fail_time': 0} for proxy in self.proxies
        }
        self._sequential_counter = 0
        if not self.proxies:
            logger.warning("ProxyManager initialized, but no proxies were provided in the configuration.")

    def _update_proxy_health(self):
        """
        Checks for unhealthy proxies whose cooldown has expired and marks them as healthy.
        """
        if not self.config.proxy_health_check_enabled:
            return

        current_time = time.time()
        for proxy, health in self.proxy_health.items():
            if health['status'] == 'unhealthy':
                if (current_time - health['last_fail_time']) > self.config.proxy_cooldown_seconds:
                    health['status'] = 'healthy'
                    health['last_fail_time'] = 0
                    logger.info(f"Proxy {proxy} has cooled down and is now marked as healthy.")

    def get_proxy(self) -> Optional[str]:
        """
        Selects a healthy proxy from the list based on the configured rotation strategy.
        It first updates the health status of proxies based on the cooldown period.
        Returns None if no healthy proxies are available.
        """
        self._update_proxy_health() # Check for cooled-down proxies first

        healthy_proxies = [p for p, health in self.proxy_health.items() if health['status'] == 'healthy']

        if not healthy_proxies:
            logger.warning("No healthy proxies available to select from.")
            return None

        strategy = self.config.proxy_rotation_strategy
        
        if strategy == 'random':
            return random.choice(healthy_proxies)
        
        elif strategy == 'sequential':
            proxy = healthy_proxies[self._sequential_counter % len(healthy_proxies)]
            self._sequential_counter += 1
            return proxy
            
        else:
            logger.warning(f"Unknown proxy rotation strategy: '{strategy}'. Defaulting to random.")
            return random.choice(healthy_proxies)

    def report_failure(self, proxy_url: str):
        """
        Marks a proxy as unhealthy after a connection failure.
        """
        if proxy_url in self.proxy_health:
            if self.config.proxy_health_check_enabled:
                self.proxy_health[proxy_url]['status'] = 'unhealthy'
                self.proxy_health[proxy_url]['last_fail_time'] = time.time()
                logger.warning(f"Proxy {proxy_url} reported as failed and marked as unhealthy.")
            else:
                logger.debug(f"Proxy failure reported for {proxy_url}, but health checks are disabled.")
        else:
            logger.warning(f"Attempted to report failure for a non-existent proxy: {proxy_url}")