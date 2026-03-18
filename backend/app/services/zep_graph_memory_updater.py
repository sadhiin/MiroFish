"""
Zep Graph Memory Update Service
Dynamically update Agent activities in the simulation to the Zep graph
"""

import os
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.zep_graph_memory_updater')


@dataclass
class AgentActivity:
    """Agent activity record"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str
    
    def to_episode_text(self) -> str:
        """
        Convert activity to a text description that can be sent to Zep
        
        Use a natural language description format to allow Zep to extract entities and relationships
        Do not add simulation-related prefixes to avoid misleading graph updates
        """
        # Generate different descriptions based on different action types
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }
        
        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()
        
        # Directly return in the format "agent name: activity description", without adding simulation prefixes
        return f"{self.agent_name}: {description}"
    
    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"posted a post: \"{content}\""
        return "posted a post"
    
    def _describe_like_post(self) -> str:
        """Liked post - includes original post content and author information"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f"liked {post_author}'s post: \"{post_content}\""
        elif post_content:
            return f"liked a post: \"{post_content}\""
        elif post_author:
            return f"liked a post by {post_author}"
        return "liked a post"
    
    def _describe_dislike_post(self) -> str:
        """Disliked post - includes original post content and author information"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f"disliked {post_author}'s post: \"{post_content}\""
        elif post_content:
            return f"disliked a post: \"{post_content}\""
        elif post_author:
            return f"disliked a post by {post_author}"
        return "disliked a post"
    
    def _describe_repost(self) -> str:
        """Reposted post - includes original post content and author information"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        
        if original_content and original_author:
            return f"Forwarded {original_author}'s post: \"{original_content}\""
        elif original_content:
            return f"Forwarded a post: \"{original_content}\""
        elif original_author:
            return f"Forwarded a post by {original_author}"
        return "Forwarded a post"
    
    def _describe_quote_post(self) -> str:
        """Quote post - includes original post content, author information, and quoted comment"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        
        base = ""
        if original_content and original_author:
            base = f"Quoted {original_author}'s post \"{original_content}\""
        elif original_content:
            base = f"Quoted a post \"{original_content}\""
        elif original_author:
            base = f"Quoted a post by {original_author}"
        else:
            base = "Quoted a post"
        
        if quote_content:
            base += f", and commented: \"{quote_content}\""
        return base
    
    def _describe_follow(self) -> str:
        """Follow user - includes the name of the followed user"""
        target_user_name = self.action_args.get("target_user_name", "")
        
        if target_user_name:
            return f"Followed user \"{target_user_name}\""
        return "Followed a user"
    
    def _describe_create_comment(self) -> str:
        """Post comment - includes comment content and the post information being commented on"""
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if content:
            if post_content and post_author:
                return f"Commented on {post_author}'s post \"{post_content}\": \"{content}\""
            elif post_content:
                return f"Commented on post \"{post_content}\": \"{content}\""
            elif post_author:
                return f"Commented on {post_author}'s post: \"{content}\""
            return f"Commented: \"{content}\""
        return "Posted a comment"
    
    def _describe_like_comment(self) -> str:
        """Like comment - includes comment content and author information"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f"Liked {comment_author}'s comment: \"{comment_content}\""
        elif comment_content:
            return f"Liked a comment: \"{comment_content}\""
        elif comment_author:
            return f"Liked a comment by {comment_author}"
        return "Liked a comment"
    
    def _describe_dislike_comment(self) -> str:
        """Dislike comment - includes comment content and author information"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f"Disliked {comment_author}'s comment: \"{comment_content}\""
        elif comment_content:
            return f"Disliked a comment: \"{comment_content}\""
        elif comment_author:
            return f"Disliked a comment by {comment_author}"
        return "Disliked a comment"
    
    def _describe_search(self) -> str:
        """Search post - includes search keywords"""
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"Searched for \"{query}\"" if query else "Performed a search"
    
    def _describe_search_user(self) -> str:
        """Search user - includes search keywords"""
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"Searched for user \"{query}\"" if query else "Searched for a user"
    
    def _describe_mute(self) -> str:
        """Mute user - includes the name of the muted user"""
        target_user_name = self.action_args.get("target_user_name", "")
        
        if target_user_name:
            return f"Muted user \"{target_user_name}\""
        return "Muted a user"
    
    def _describe_generic(self) -> str:
        # Generate a generic description for unknown action types
        return f"Performed {self.action_type} operation"
class ZepGraphMemoryUpdater:
    """
    Zep Graph Memory Updater
    
    Monitors the simulated actions log file and updates new agent activities to the Zep graph in real-time.
    Groups by platform and sends in batches to Zep after accumulating BATCH_SIZE activities.
    
    All meaningful behaviors will be updated to Zep, and action_args will contain complete context information:
    - Original post of liked/disliked posts
    - Original post of forwarded/quoted posts
    - Username of followed/blocked users
    - Original comment of liked/disliked comments
    """
    
    # Batch send size (send after accumulating a certain number per platform)
    BATCH_SIZE = 5
    
    # Platform name mapping (for console display)
    PLATFORM_DISPLAY_NAMES = {
        'twitter': 'World 1',
        'reddit': 'World 2',
    }
    
    # Send interval (seconds), to avoid requests being too fast
    SEND_INTERVAL = 0.5
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        """
        Initialize updater
        
        Args:
            graph_id: Zep graph ID
            api_key: Zep API Key (optional, defaults to reading from config)
        """
        self.graph_id = graph_id
        self.api_key = api_key or Config.ZEP_API_KEY
        
        if not self.api_key:
            raise ValueError("ZEP_API_KEY not configured")
        
        self.client = Zep(api_key=self.api_key)
        
        # Activity queue
        self._activity_queue: Queue = Queue()
        
        # Activity buffer grouped by platform (each platform accumulates to BATCH_SIZE before batch sending)
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()
        
        # Control flags
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._total_activities = 0  # Number of activities actually added to the queue
        self._total_sent = 0        # Number of batches successfully sent to Zep
        self._total_items_sent = 0  # Number of activities successfully sent to Zep
        self._failed_count = 0      # Number of batches failed to send
        self._skipped_count = 0     # Number of activities skipped (DO_NOTHING)
        
        logger.info(f"ZepGraphMemoryUpdater initialized: graph_id={graph_id}, batch_size={self.BATCH_SIZE}")
    
    def _get_platform_display_name(self, platform: str) -> str:
        """Get the display name of the platform"""
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)
    
    def start(self):
        """Start the background worker thread"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"ZepMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"ZepGraphMemoryUpdater started: graph_id={self.graph_id}")
    
    def stop(self):
        """Stop the background worker thread"""
        self._running = False
        
        # Send remaining activities
        self._flush_remaining()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        
        logger.info(f"ZepGraphMemoryUpdater stopped: graph_id={self.graph_id}, "
                   f"total_activities={self._total_activities}, "
                   f"batches_sent={self._total_sent}, "
                   f"items_sent={self._total_items_sent}, "
                   f"failed={self._failed_count}, "
                   f"skipped={self._skipped_count}")
    
    def add_activity(self, activity: AgentActivity):
        """
        Add an agent activity to the queue
        
        All meaningful actions will be added to the queue, including:
        - CREATE_POST (post)
        - CREATE_COMMENT (comment)
        - QUOTE_POST (quote post)
        - SEARCH_POSTS (search posts)
        - SEARCH_USER (search user)
        - LIKE_POST/DISLIKE_POST (like/dislike post)
        - REPOST (repost)
        - FOLLOW (follow)
        - MUTE (mute)
        - LIKE_COMMENT/DISLIKE_COMMENT (like/dislike comment)
        
        action_args will contain complete context information (such as original post, username, etc.).
        
        Args:
            activity: Agent activity record
        """
        # Skip DO_NOTHING type activities
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return
        
        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(f"Add activity to Zep queue: {activity.agent_name} - {activity.action_type}")
    
    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """
        Add activity from dictionary data
        
        Args:
            data: Dictionary data parsed from actions.jsonl
            platform: Platform name (twitter/reddit)
        """
        # Skip entries of event type
        if "event_type" in data:
            return
        
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        
        self.add_activity(activity)
    
    def _worker_loop(self):
        """Background work loop - Send activities to Zep in batches by platform"""
        while self._running or not self._activity_queue.empty():
            try:
                # Try to get activity from queue (timeout 1 second)
                try:
                    activity = self._activity_queue.get(timeout=1)
                    
                    # Add activity to the corresponding platform buffer
                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)
                        
                        # Check if the platform has reached batch size
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE:]
                            # Release lock before sending
                            self._send_batch_activities(batch, platform)
                            # Send interval to avoid requests too fast
                            time.sleep(self.SEND_INTERVAL)
                    
                except Empty:
                    pass
                    
            except Exception as e:
                logger.error(f"Work loop exception: {e}")
                time.sleep(1)
    
    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        Send activities to Zep graph in batches (merged into one text)
        
        Args:
            activities: List of agent activities
            platform: Platform name
        """
        if not activities:
            return
        # Merge multiple activities into one text, separated by newlines
        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)
        
        # Send with retries
        for attempt in range(self.MAX_RETRIES):
            try:
                self.client.graph.add(
                    graph_id=self.graph_id,
                    type="text",
                    data=combined_text
                )
                
                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(f"Successfully sent {len(activities)} {display_name} activities to graph {self.graph_id}")
                logger.debug(f"Batch content preview: {combined_text[:200]}...")
                return
                
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Batch send to Zep failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Batch send to Zep failed, retried {self.MAX_RETRIES} times: {e}")
                    self._failed_count += 1
    
    def _flush_remaining(self):
        """Send remaining activities in the queue and buffer"""
        # First process remaining activities in the queue, add to buffer
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break
        
        # Then send remaining activities in each platform buffer (even if less than BATCH_SIZE)
        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    display_name = self._get_platform_display_name(platform)
                    logger.info(f"Sending remaining {len(buffer)} activities on {display_name} platform")
                    self._send_batch_activities(buffer, platform)
            # Clear all buffers
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}
        
        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,  # Total number of activities added to the queue
            "batches_sent": self._total_sent,            # Number of batches successfully sent
            "items_sent": self._total_items_sent,        # Number of activities successfully sent
            "failed_count": self._failed_count,          # Number of batches failed to send
            "skipped_count": self._skipped_count,        # Number of activities skipped (DO_NOTHING)
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,                # Size of each platform buffer
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """
    Manage multiple simulated Zep graph memory updaters
    
    Each simulation can have its own updater instance
    """
    
    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()
    
    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """
        Create a graph memory updater for a simulation
        
        Args:
            simulation_id: Simulation ID
            graph_id: Zep graph ID
            
        Returns:
            ZepGraphMemoryUpdater instance
        """
        with cls._lock:
            # If it already exists, stop the old one first
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            
            updater = ZepGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater
            
            logger.info(f"Created graph memory updater: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater
    
    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """Get the updater for the simulation"""
        return cls._updaters.get(simulation_id)
    
    @classmethod
    def stop_updater(cls, simulation_id: str):
        """Stop and remove the updater for the simulation"""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info(f"Stopped graph memory updater: simulation_id={simulation_id}")
    
    # Flag to prevent stop_all from being called multiple times
    _stop_all_done = False
    
    @classmethod
    def stop_all(cls):
        """Stop all updaters"""
        # Prevent multiple calls
        if cls._stop_all_done:
            return
        cls._stop_all_done = True
        
        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as e:
                        logger.error(f"Failed to stop updater: simulation_id={simulation_id}, error={e}")
                cls._updaters.clear()
            logger.info("Stopped all graph memory updaters")
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all updaters"""
        return {
            sim_id: updater.get_stats() 
            for sim_id, updater in cls._updaters.items()
        }
