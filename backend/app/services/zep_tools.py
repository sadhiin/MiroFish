"""
Zep retrieval tool service
Encapsulates tools for graph search, node reading, edge querying, etc., for use by Report Agent

Core retrieval tools (optimized):
1. InsightForge (deep insight retrieval) - The most powerful hybrid retrieval, automatically generates sub-questions and retrieves in multiple dimensions
2. PanoramaSearch (breadth search) - Get the full picture, including expired content
3. QuickSearch (simple search) - Quick retrieval
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """Search result"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """Convert to text format for LLM understanding"""
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} related information"]
        
        if self.facts:
            text_parts.append("\n### Related facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node information"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """Convert to text format"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unknown type")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge information"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Time information
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relationship: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "Unknown"
            invalid_at = self.invalid_at or "Present"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expired: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """Is expired"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """Is invalid"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Deep Insight Retrieval Result (InsightForge)
    Contains retrieval results of multiple sub-questions and comprehensive analysis
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # Retrieval results for each dimension
    semantic_facts: List[str] = field(default_factory=list)  # Semantic search results
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Entity insights
    relationship_chains: List[str] = field(default_factory=list)  # Relationship chains
    
    # Statistics
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """Convert to detailed text format for LLM understanding"""
        text_parts = [
            f"## Future Prediction Deep Analysis",
            f"Analysis Question: {self.query}",
            f"Prediction Scenario: {self.simulation_requirement}",
            f"\n### Prediction Data Statistics",
            f"- Related Prediction Facts: {self.total_facts} items",
            f"- Involved Entities: {self.total_entities} items",
            f"- Relationship Chains: {self.total_relationships} items"
        ]
        
        # Sub-questions
        if self.sub_queries:
            text_parts.append(f"\n### Analyzed Sub-questions")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # Semantic search results
        if self.semantic_facts:
            text_parts.append(f"\n### 【Key Facts】(Please cite these originals in the report)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Entity insights
        if self.entity_insights:
            text_parts.append(f"\n### 【Core Entities】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Summary: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Related facts: {len(entity.get('related_facts', []))} items")
        
        # Relationship chain
        if self.relationship_chains:
            text_parts.append(f"\n### 【Relationship chain】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Breadth search result (Panorama)
    Contains all relevant information, including expired content
    """
    query: str
    
    # All nodes
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # All edges (including expired ones)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Currently active facts
    active_facts: List[str] = field(default_factory=list)
    # Expired/invalid facts (historical records)
    historical_facts: List[str] = field(default_factory=list)
    
    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """Convert to text format (full version, not truncated)"""
        text_parts = [
            f"## Breadth search result (Future panorama view)",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total number of nodes: {self.total_nodes}",
            f"- Total number of edges: {self.total_edges}",
            f"- Currently active facts: {self.active_count} items",
            f"- Historical/expired facts: {self.historical_count} items"
        ]
        
        # Currently active facts (full output, not truncated)
        if self.active_facts:
            text_parts.append(f"\n### 【Currently active facts】(Simulated result original text)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Historical/expired facts (full output, not truncated)
        if self.historical_facts:
            text_parts.append(f"\n### 【Historical/expired facts】(Evolution process record)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Key entities (full output, not truncated)
        if self.all_nodes:
            text_parts.append(f"\n### 【Involved entities】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Interview results of a single Agent"""
    agent_name: str
    agent_role: str  # Role type (e.g., student, teacher, media, etc.)
    agent_bio: str  # Biography
    question: str  # Interview question
    response: str  # Interview response
    key_quotes: List[str] = field(default_factory=list)  # Key quotes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Display the full agent_bio, do not truncate
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key Quotes:**\n"
            for quote in self.key_quotes:
                # Clean up various quotes
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # Remove leading punctuation
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Filter out junk content containing question numbers (questions 1-9)
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Truncate overly long content (truncate by period, not hard truncate)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interview results
    Contains interview responses from multiple simulated Agents
    """
    interview_topic: str  # Interview topic
    interview_questions: List[str]  # List of interview questions
    
    # Selected Agents for the interview
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Interview responses from each Agent
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # Reasoning for selecting Agents
    selection_reasoning: str = ""
    # Consolidated interview summary
    summary: str = ""
    
    # Statistics
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """Convert to detailed text format for LLM understanding and report reference"""
        text_parts = [
            "## In-depth Interview Report",
            f"**Interview Topic:** {self.interview_topic}",
            f"**Number of Interviews:** {self.interviewed_count} / {self.total_agents} simulated Agents",
            "\n### Reason for Selecting Interviewees",
            self.selection_reasoning or "(Automatically selected)",
            "\n---",
            "\n### Interview Transcript",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        text_parts.append("\n### Interview Summary and Core Points")
        text_parts.append(self.summary or "(No summary)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep Retrieval Tool Service
    
    [Core Retrieval Tools - Optimized]
    1. insight_forge - Deep Insight Retrieval (most powerful, automatically generates sub-questions, multi-dimensional retrieval)
    2. panorama_search - Broad Search (get the full picture, including expired content)
    3. quick_search - Simple Search (quick retrieval)
    4. interview_agents - In-depth Interview (interview simulation Agent, obtain multi-perspective views)
    
    [Basic Tools]
    - search_graph - Graph Semantic Search
    - get_all_nodes - Get all nodes of the graph
    - get_all_edges - Get all edges of the graph (including time information)
    - get_node_detail - Get detailed information of a node
    - get_node_edges - Get edges related to a node
    - get_entities_by_type - Get entities by type
    - get_entity_summary - Get a summary of entity relationships
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY not configured")
        
        self.client = Zep(api_key=self.api_key)
        # LLM client used for InsightForge to generate sub-questions
        self._llm_client = llm_client
        logger.info("ZepToolsService initialization complete")
    
    @property
    def llm(self) -> LLMClient:
        """Lazy initialization of LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """API call with retry mechanism"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} attempt {attempt + 1} failed: {str(e)[:100]}, "
                        f"retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} failed after {max_retries} attempts: {str(e)}")
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Graph Semantic Search
        
        Use hybrid search (semantic + BM25) to search for relevant information in the graph.
        If the search API of Zep Cloud is unavailable, fallback to local keyword matching.
        
        Args:
            graph_id: Graph ID (Standalone Graph)
            query: Search query
            limit: Number of results to return
            scope: Search scope, "edges" or "nodes"
            
        Returns:
            SearchResult: Search results
        """
        logger.info(f"Graph search: graph_id={graph_id}, query={query[:50]}...")
        
        # Attempt to use Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"Graph search(graph={graph_id})"
            )
            
            facts = []
            edges = []
            nodes = []
            
            # Parse edge search results
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # Parse node search results
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # Node summary is also considered a fact
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"Search completed: Found {len(facts)} related facts")
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(f"Zep Search API failed, downgraded to local search: {str(e)}")
            # Downgrade: Use local keyword matching search
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Local keyword matching search (as a downgrade solution for Zep Search API)
        
        Get all edges/nodes, then perform keyword matching locally
        
        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return
            scope: Search scope
            
        Returns:
            SearchResult: Search results
        """
        logger.info(f"Using local search: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # Extract query keywords (simple tokenization)
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """Calculate the match score between text and query"""
            if not text:
                return 0
            text_lower = text.lower()
            # Exact match query
            if query_lower in text_lower:
                return 100
            # Keyword match
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Get all edges and match
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                # Sort by score
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Get all nodes and match
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"Local search completed: Found {len(facts)} related facts")
            
        except Exception as e:
            logger.error(f"Local search failed: {str(e)}")
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Get all nodes of the graph (paged retrieval)

        Args:
            graph_id: Graph ID

        Returns:
            Node list
        """
        logger.info(f"Getting all nodes of graph {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"Retrieved {len(result)} nodes")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        Get all edges of the graph (paged retrieval, includes temporal information)

        Args:
            graph_id: Graph ID
            include_temporal: Whether to include temporal information (default True)

        Returns:
            Edge list (includes created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(f"Getting all edges of graph {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # Add time information
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"Retrieved {len(result)} edges")
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Get detailed information of a single node
        
        Args:
            node_uuid: Node UUID
            
        Returns:
            Node information or None
        """
        logger.info(f"Retrieving node details: {node_uuid[:8]}...")
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"Retrieving node details(uuid={node_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"Failed to retrieve node details: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Get all edges related to a node
        
        By retrieving all edges of the graph, then filtering out the edges related to the specified node
        
        Args:
            graph_id: Graph ID
            node_uuid: Node UUID
            
        Returns:
            List of edges
        """
        logger.info(f"Retrieving edges related to node {node_uuid[:8]}...")
        
        try:
            # Retrieve all edges of the graph, then filter
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # Check if the edge is related to the specified node (as source or target)
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(f"Found {len(result)} edges related to the node")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to retrieve node edges: {str(e)}")
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        Get entities by type
        
        Args:
            graph_id: Graph ID
            entity_type: Entity type (e.g., Student, PublicFigure, etc.)
            
        Returns:
            List of entities matching the type
        """
        logger.info(f"Retrieving entities of type {entity_type}...")
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Check if labels contain the specified type
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(f"Found {len(filtered)} entities of type {entity_type}")
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get the relationship summary of the specified entity
        
        Search for all information related to the entity and generate a summary
        
        Args:
            graph_id: Graph ID
            entity_name: Entity name
            
        Returns:
            Entity summary information
        """
        logger.info(f"Getting relationship summary for entity {entity_name}...")
        
        # First search for information related to the entity
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # Try to find the entity among all nodes
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # Pass in graph_id parameter
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get statistics of the graph
        
        Args:
            graph_id: Graph ID
            
        Returns:
            Statistics
        """
        logger.info(f"Getting statistics for graph {graph_id}...")
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # Count the distribution of entity types
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # Count the distribution of relation types
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Get context information related to the simulation
        
        Comprehensive search for all information related to simulation requirements
        
        Args:
            graph_id: Graph ID
            simulation_requirement: Simulation requirement description
            limit: Limit on the number of each type of information
            
        Returns:
            Simulation context information
        """
        logger.info(f"Getting simulation context: {simulation_requirement[:50]}...")
        
        # Search for information related to the simulation requirement
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Get graph statistics
        stats = self.get_graph_statistics(graph_id)
        
        # Get all entity nodes
        all_nodes = self.get_all_nodes(graph_id)
        
        # Filter entities with actual types (non-pure Entity nodes)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Limit the number
            "total_entities": len(entities)
        }
    
    # ========== Core Retrieval Tool (Optimized) ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        【InsightForge - Deep Insight Retrieval】
        
        The most powerful hybrid retrieval function, automatically decomposes problems and retrieves in multiple dimensions:
        1. Use LLM to decompose the problem into multiple sub-problems
        2. Perform semantic search for each sub-problem
        3. Extract related entities and obtain their detailed information
        4. Trace relationship chains
        5. Integrate all results to generate deep insights
        
        Args:
            graph_id: Graph ID
            query: User question
            simulation_requirement: Simulation requirement description
            report_context: Report context (optional, for more precise sub-problem generation)
            max_sub_queries: Maximum number of sub-problems
            
        Returns:
            InsightForgeResult: Deep insight retrieval result
        """
        logger.info(f"InsightForge Deep Insight Retrieval: {query[:50]}...")
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: Use LLM to generate sub-problems
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"Generated {len(sub_queries)} sub-problems")
        
        # Step 2: Perform semantic search for each sub-problem
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # Also search the original question
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: Extract relevant entity UUIDs from edges, only get information of these entities (do not get all nodes)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Get details of all relevant entities (no limit on quantity, full output)
        entity_insights = []
        node_map = {}  # Used for subsequent relationship chain construction
        
        for uuid in list(entity_uuids):  # Process all entities, do not truncate
            if not uuid:
                continue
            try:
                # Get information of each relevant node individually
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                    
                    # Get all facts related to this entity (do not truncate)
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # Full output, do not truncate
                    })
            except Exception as e:
                logger.debug(f"Failed to get node {uuid}: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Construct all relationship chains (no limit on quantity)
        relationship_chains = []
        for edge_data in all_edges:  # Process all edges, do not truncate
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(f"InsightForge completed: {result.total_facts} facts, {result.total_entities} entities, {result.total_relationships} relationships")
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        Use LLM to generate sub-questions
        Break down complex problems into multiple sub-problems that can be independently retrieved
        """
        system_prompt = """You are a professional problem analysis expert. Your task is to break down a complex problem into multiple sub-problems that can be independently observed in a simulated world.

Requirements:
1. Each sub-problem should be specific enough to find related Agent behaviors or events in the simulated world
2. Sub-problems should cover different dimensions of the original problem (e.g.: who, what, why, how, when, where)
3. Sub-problems should be relevant to the simulation scenario
4. Return in JSON format: {"sub_queries": ["Sub-problem 1", "Sub-problem 2", ...]}"""

        user_prompt = f"""Simulation requirement background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please break down the following problem into {max_queries} sub-problems:
{query}

Return a list of sub-problems in JSON format."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # Ensure it is a list of strings
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(f"Failed to generate sub-problems: {str(e)}, using default sub-problems")
            # Fallback: return variants based on the original problem
            return [
                query,
                f"Main participants of {query}",
                f"Reasons and impacts of {query}",
                f"Development process of {query}"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        [PanoramaSearch - Breadth Search]
        
        Get a panoramic view, including all relevant content and historical/expired information:
        1. Get all relevant nodes
        2. Get all edges (including expired/invalid ones)
        3. Classify and organize current valid and historical information
        
        This tool is suitable for scenarios where you need to understand the full picture of an event and track its evolution.
        
        Args:
            graph_id: Graph ID
            query: Search query (for relevance ranking)
            include_expired: Whether to include expired content (default True)
            limit: Limit on the number of returned results
            
        Returns:
            PanoramaResult: Breadth search result
        """
        logger.info(f"PanoramaSearch Breadth Search: {query[:50]}...")
        
        result = PanoramaResult(query=query)
        
        # Get all nodes
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Get all edges (including temporal information)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # Classify facts
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # Add entity names to facts
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # Determine if expired/invalid
            is_historical = edge.is_expired or edge.is_invalid
            if is_historical:
                # Historical/expired fact, add time stamp
                valid_at = edge.valid_at or "Unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "Unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Currently valid fact
                active_facts.append(edge.fact)
        
        # Relevance sorting based on query
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Sort and limit quantity
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(f"PanoramaSearch completed: {result.active_count} active, {result.historical_count} historical")
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        [QuickSearch - Simple Search]
        
        Fast, lightweight retrieval tool:
        1. Directly call Zep semantic search
        2. Return the most relevant results
        3. Suitable for simple, direct retrieval needs
        
        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return
            
        Returns:
            SearchResult: Search results
        """
        logger.info(f"QuickSearch simple search: {query[:50]}...")
        
        # Directly call the existing search_graph method
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(f"QuickSearch completed: {result.total_count} results")
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        [InterviewAgents - In-depth Interview]
        
        Call the real OASIS interview API to interview agents running in the simulation:
        1. Automatically read persona files to understand all simulated agents
        2. Use LLM to analyze interview requirements and intelligently select the most relevant agents
        3. Use LLM to generate interview questions
        4. Call the /api/simulation/interview/batch interface for real interviews (simultaneous interviews on both platforms)
        5. Integrate all interview results to generate an interview report
        
        [Important] This feature requires the simulation environment to be running (OASIS environment not closed)
        
        [Usage Scenarios]
        - Need to understand event perspectives from different roles
        - Need to collect opinions and viewpoints from multiple parties
        - Need to obtain real answers from simulated agents (not LLM simulated)
        
        Args:
            simulation_id: Simulation ID (used to locate persona files and call the interview API)
            interview_requirement: Interview requirement description (unstructured, e.g., "Understand students' views on the event")
            simulation_requirement: Simulation requirement background (optional)
            max_agents: Maximum number of agents to interview
            custom_questions: Custom interview questions (optional, automatically generated if not provided)
            
        Returns:
            InterviewResult: Interview result
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(f"InterviewAgents in-depth interview (real API): {interview_requirement[:50]}...")
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: Read agent profile files
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            logger.warning(f"No profile files found for simulation {simulation_id}")
            result.summary = "No agent profile files available for interview"
            return result
        
        result.total_agents = len(profiles)
        logger.info(f"Loaded {len(profiles)} agent profiles")
        
        # Step 2: Use LLM to select agents for interview (returns a list of agent_ids)
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"Selected {len(selected_agents)} agents for interview: {selected_indices}")
        
        # Step 3: Generate interview questions (if not provided)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"Generated {len(result.interview_questions)} interview questions")
        
        # Combine questions into a single interview prompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # Add optimization prefix to constrain agent response format
        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Please combine your profile, all past memories, and actions, "
            "and answer the following questions directly in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language, do not invoke any tools\n"
            "2. Do not return in JSON format or tool invocation format\n"
            "3. Do not use Markdown headings (e.g., #, ##, ###)\n"
            "4. Answer each question in order, starting each answer with 'Question X:' (X is the question number)\n"
            "5. Separate each answer with a blank line\n"
            "6. Answers should be substantive, with at least 2-3 sentences per question\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: Call the real interview API (platform not specified, default to dual-platform interview)
        try:
            # Construct batch interview list (platform not specified, dual-platform interview)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # Use the optimized prompt
                    # Platform not specified, API will interview on both Twitter and Reddit
                })
            
            logger.info(f"Calling batch interview API (dual-platform): {len(interviews_request)} agents")
            
            # Call the batch interview method of SimulationRunner (platform not passed, dual-platform interview)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # Platform not specified, dual-platform interview
                timeout=180.0   # Dual-platform requires longer timeout
            )
            
            logger.info(f"Interview API returned: {api_result.get('interviews_count', 0)} results, success={api_result.get('success')}")
            
            # Check if API call was successful
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unknown error")
                logger.warning(f"Interview API returned failure: {error_msg}")
                result.summary = f"Interview API call failed: {error_msg}. Please check the OASIS simulation environment status."
                return result
            
            # Step 5: Parse API return results, construct AgentInterview object
            # Dual-platform mode return format: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")
                
                # Get the interview results of the Agent on two platforms
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Clean possible tool call JSON wrapping
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Always output dual-platform tags
                twitter_text = twitter_response if twitter_response else "(No response received on this platform)"
                reddit_text = reddit_response if reddit_response else "(No response received on this platform)"
                response_text = f"[Twitter Platform Response]\n{twitter_text}\n\n[Reddit Platform Response]\n{reddit_text}"

                # Extract key quotes (from responses on both platforms)
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Clean response text: remove tags, numbers, Markdown, etc.
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'Question\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'\[[^\]]+\]', '', clean_text)

                # Strategy 1 (Main): Extract complete sentences with substantial content
                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', 'Question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "." for s in meaningful[:3]]

                # Strategy 2 (Supplementary): Correctly paired long text within Chinese 
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # Expand bio length limit
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # Simulation environment not running
            logger.warning(f"Interview API call failed (environment not running?): {e}")
            result.summary = f"Interview failed: {str(e)}. The simulation environment may be closed, please ensure the OASIS environment is running."
            return result
        except Exception as e:
            logger.error(f"Interview API call exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"An error occurred during the interview process: {str(e)}"
            return result
        
        # Step 6: Generate interview summary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(f"InterviewAgents completed: Interviewed {result.interviewed_count} Agents (dual-platform)")
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Clean JSON tool call wrapping in Agent responses, extract actual content"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load simulated Agent profile files"""
        import os
        import csv
        
        # Construct profile file path
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # First try to read Reddit JSON format
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"Loaded {len(profiles)} profiles from reddit_profiles.json")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read reddit_profiles.json: {e}")
        
        # Try to read Twitter CSV format
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert CSV format to unified format
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown"
                        })
                logger.info(f"Loaded {len(profiles)} profiles from twitter_profiles.csv")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read twitter_profiles.csv: {e}")
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        Use LLM to select Agents for interview
        
        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: List of complete information of selected Agents
                - selected_indices: List of indices of selected Agents (for API calls)
                - reasoning: Reason for selection
        """
        
        # Construct Agent summary list
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """You are a professional interview planning expert. Your task is to select the most suitable subjects for interview from the list of simulated Agents based on the interview requirements.

Selection criteria:
1. The Agent's identity/profession is related to the interview topic
2. The Agent may hold unique or valuable perspectives
3. Select diverse perspectives (e.g., supporters, opponents, neutrals, professionals, etc.)
4. Prefer roles directly related to the event

Return in JSON format:
{
    "selected_indices": [List of indices of selected Agents],
    "reasoning": "Explanation of selection reasoning"
}"""

        user_prompt = f"""Interview requirements:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else "Not provided"}

List of selectable Agents (total {len(agent_summaries)}):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select up to {max_agents} agents most suitable for the interview and explain the reasons for your choice."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Automatically selected based on relevance")
            
            # Get complete information of selected agents
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(f"LLM failed to select agents, using default selection: {e}")
            # Fallback: select the first N
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Using default selection strategy"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate interview questions using LLM"""
        
        agent_roles = [a.get("profession", "Unknown") for a in selected_agents]
        
        system_prompt = """You are a professional journalist/interviewer. Based on the interview requirements, generate 3-5 in-depth interview questions.

Question requirements:
1. Open-ended questions that encourage detailed answers
2. Different roles may have different answers
3. Cover multiple dimensions such as facts, opinions, feelings, etc.
4. Natural language, like a real interview
5. Each question should be within 50 words, concise and clear
6. Ask directly, do not include background explanations or prefixes

Return in JSON format: {"questions": ["Question 1", "Question 2", ...]}"""

        user_prompt = f"""Interview requirement: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else "Not provided"}

Interviewee roles: {', '.join(agent_roles)}

Please generate 3-5 interview questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [f"What are your views on {interview_requirement}?"])
            
        except Exception as e:
            logger.warning(f"Failed to generate interview questions: {e}")
            return [
                f"What is your opinion on {interview_requirement}?",
                "What impact does this have on you or the group you represent?",
                "How do you think this issue should be resolved or improved?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Generate interview summary"""
        
        if not interviews:
            return "No interviews completed"
        
        # Collect all interview content
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}")
        
        system_prompt = """You are a professional news editor. Please generate an interview summary based on the responses of multiple interviewees.

Summary requirements:
1. Extract the main viewpoints of each party
2. Point out the consensus and differences of the viewpoints
3. Highlight valuable quotes
4. Be objective and neutral, do not favor any side
5. Keep it within 1000 words

Format constraints (must be followed):
- Use plain text paragraphs, separate different sections with blank lines
- Do not use Markdown headers (such as #, ##, ###)
- Do not use separators (such as ---, ***)
- Use Chinese quotation marks 「」 when quoting respondents' original words
- You can use **bold** to mark keywords, but do not use other Markdown syntax"""

        user_prompt = f"""Interview topic: {interview_requirement}

Interview content:
{"".join(interview_texts)}

Please generate an interview summary."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate interview summary: {e}")
            # Fallback: simple concatenation
            return f"Interviewed a total of {len(interviews)} respondents, including: " + "、".join([i.agent_name for i in interviews])
