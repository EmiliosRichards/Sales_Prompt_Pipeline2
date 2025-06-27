import logging
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from ..core.config import AppConfig

logger = logging.getLogger(__name__)

class InteractionHandler:
    def __init__(self, page: Page, config: AppConfig):
        self.page = page
        self.config = config

    async def handle_interactions(self) -> bool:
        """
        Performs a limited number of quick passes to close modal dialogs,
        cookie banners, etc., based on configured selectors and text queries.
        Returns True if any interaction was successfully handled.
        """
        if not self.config.interaction_handler_enabled:
            logger.debug("Interaction handler is disabled in the configuration.")
            return False

        max_passes = self.config.interaction_handler_max_passes
        visibility_timeout = self.config.interaction_handler_visibility_timeout_ms
        any_interaction_handled = False

        for i in range(max_passes):
            handled_in_pass = False
            interactions = [("selector", s) for s in self.config.interaction_selectors] + \
                           [("text", t) for t in self.config.interaction_text_queries]

            for type, query in interactions:
                try:
                    element = None
                    if type == "selector":
                        element = self.page.locator(query).first
                    else: # text
                        element = self.page.locator(f"*:visible:text-matches('{query}', 'i')").first
                    
                    if await element.is_visible(timeout=visibility_timeout):
                        logger.info(f"Found and clicking element by {type}: '{query}'")
                        await element.click(timeout=1000)
                        handled_in_pass = True
                        any_interaction_handled = True
                        await self.page.wait_for_timeout(500) # wait for UI to settle
                        break # break from the for loop to restart the pass
                except PlaywrightTimeoutError:
                    # This is expected if the element is not visible within the short timeout
                    logger.debug(f"Element not visible or timed out for {type} '{query}'.")
                except Exception as e:
                    logger.warning(f"Error handling {type} '{query}': {e}")
            
            if not handled_in_pass:
                # If a full pass completes with no interactions handled, we can exit early.
                logger.debug(f"No interactive elements found in pass {i+1}/{max_passes}. Exiting handler.")
                return any_interaction_handled
        
        logger.debug(f"Completed {max_passes} interaction handling passes.")
        return any_interaction_handled