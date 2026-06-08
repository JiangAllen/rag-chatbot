from neo4j import GraphDatabase, RoutingControl
from py2neo import Graph, Node, NodeMatcher, Relationship
import pandas as pd
import math

class Neo4jConnection:
    def __init__(self, url, auth):
        try:
            self.driver = GraphDatabase.driver(url, auth=auth)
            self.graph = Graph(url, auth=auth)
        except Exception as e:
            print("Failed to read nodes: {}".format(e))

    def read_nodes(self):
        try:
            result = self.graph.run(
                """
                MATCH (n)
                RETURN DISTINCT labels(n) AS labels, n.name AS name, properties(n) AS props
                """
            )
            nodes = list(result)
            for node in nodes:
                print(node)
            print("Returned {} nodes.".format(len(nodes)))
            return nodes
        except Exception as e:
            print("Failed to get all nodes: {}".format(e))
            return []

    def read_relationships(self):
        try:
            result = self.graph.run(
                """
                MATCH (a)-[r]->(b)
                RETURN a.name AS from, type(r) AS rel_type, b.name AS to, properties(r) AS props
                """
            )
            rels = list(result)
            for rel in rels:
                print(rel)
            print("Returned {} relationships.".format(len(rels)))
            return rels
        except Exception as e:
            print("Failed to get all relationships: {}".format(e))
            return []

    def find_node(self, nodename, label="Entity"):
        try:
            matcher = NodeMatcher(self.graph)
            node = matcher.match(label, name=nodename).first()
            if node:
                print("Found node: {}".format(dict(node)))
                return dict(node)
            else:
                print("Node '{}' not found.".format(nodename))
                return None
        except Exception as e:
            print("Failed to find node {}: {}".format(nodename, e))
            return None

    def create_nodes(self, jsoninput):
        for js in jsoninput:
            try:
                subj = js["subject"]
                pred = js["predicate"].replace(" ", "_").upper()
                obj = js["object"]
                node1 = Node("Entity", name=subj)
                node2 = Node("Entity", name=obj)
                self.graph.merge(node1, "Entity", "name")
                self.graph.merge(node2, "Entity", "name")
                rel = Relationship(node1, pred, node2)
                self.graph.merge(rel)
            except KeyError as e:
                print("Missing key in JSON: {}".format(e))
            except Exception as e:
                print("Error creating node/relationship: {}".format(e))
        print("Knowledge graph created/updated successfully.")

    def update_node(self, nodename, updates: dict, label="Entity"):
        try:
            matcher = NodeMatcher(self.graph)
            node = matcher.match(label, name=nodename).first()
            if node:
                for k, v in updates.items():
                    node[k] = v
                self.graph.push(node)
                print("Node '{}' updated with {}".format(nodename, updates))
            else:
                print("Node '{}' not found.".format(nodename))
        except Exception as e:
            print("Failed to update node {}: {}".format(nodename, e))

    def delete_node(self, nodename, label="Entity"):
        matcher = NodeMatcher(self.graph)
        node = matcher.match(label, name=nodename).first()
        if node:
            self.graph.separate(node)
            self.graph.delete(node)
            print(f"Deleted node: {nodename}")
        else:
            print(f"Node {nodename} not found.")

    def clear_graph(self):
        try:
            self.graph.run("MATCH (n) DETACH DELETE n")
            print("All nodes and relationships have been deleted.")
        except Exception as e:
            print("Failed to clear the graph: {}".format(e))

    def close_connection(self):
        try:
            self.driver.close()
        except Exception as e:
            print("Error closing connection: {}".format(e))

    def ingest_neo4j(self, rows: list, chunk_size: int, mode: str):
        imdb_cypher = """
        UNWIND $rows AS row
        MERGE (m:Movie {title: row.title})
        SET m.year = row.year,
            m.certificate = row.certificate,
            m.duration = row.duration,
            m.rating = row.rating,
            m.metascore = row.metascore,
            m.description = row.description
        WITH m, row
        FOREACH (gName IN row.genres |
          MERGE (g:Genre {name: gName})
          MERGE (m)-[:HAS_GENRE]->(g)
        )
        FOREACH (dName IN row.directors |
          MERGE (d:Director {name: dName})
          MERGE (m)-[:DIRECTED_BY]->(d)
        )
        FOREACH (aName IN row.cast |
          MERGE (a:Actor {name: aName})
          MERGE (m)-[:STARRING]->(a)
        )
        FOREACH (_ IN CASE WHEN row.review_text IS NULL OR row.review_text = '' OR row.review_title IS NULL OR row.review_title = '' THEN [] ELSE [1] END |
          MERGE (rev:Review {title: row.review_title})
          SET rev.text = row.review_text
          MERGE (m)-[:HAS_REVIEW]->(rev)
        )
        """
        mbti_cypher = """
        UNWIND $rows AS row
        MERGE (n:Name {name: row.name})
            ON CREATE SET 
                n.enneagram = row.enneagram,
                n.weight = row.weight,
                n.textEmbedding = row.nameEmbedding
        MERGE (p:Personality {four_letter: row.four_letter})
        MERGE (c:Category {name: row.category})
            ON CREATE SET c.textEmbedding = row.categoryEmbedding
        MERGE (sc:Subcategory {name: row.subcategory})
            ON CREATE SET sc.textEmbedding = row.subcategoryEmbedding
        MERGE (n)-[:HAS_TYPE]->(p)
        MERGE (c)-[:HAS_SUBCATEGORY]->(sc)
        MERGE (p)-[:BELONGS_TO]->(sc)
        MERGE (n)-[:INTEREST_IN]->(sc)
        """
        personality_cypher = """
        UNWIND $rows AS row
        MERGE (p:Personality {four_letter: row.four_letter})
        FOREACH (_ IN CASE WHEN row.overview IS NULL OR row.overview = '' THEN [] ELSE [1] END |
          MERGE (o:Overview {text: row.overview})
          MERGE (p)-[:HAS_OVERVIEW]->(o)
        )
        FOREACH (_ IN CASE WHEN row.strengths IS NULL OR row.strengths = '' THEN [] ELSE [1] END |
          MERGE (s:Strength {text: row.strengths})
          MERGE (p)-[:HAS_STRENGTH]->(s)
        )
        FOREACH (_ IN CASE WHEN row.weaknesses IS NULL OR row.weaknesses = '' THEN [] ELSE [1] END |
          MERGE (w:Weakness {text: row.weaknesses})
          MERGE (p)-[:HAS_WEAKNESS]->(w)
        )
        FOREACH (_ IN CASE WHEN row.relationships IS NULL OR row.relationships = '' THEN [] ELSE [1] END |
          MERGE (r:Relationship {text: row.relationships})
          MERGE (p)-[:HAS_RELATIONSHIP]->(r)
        )
        FOREACH (_ IN CASE WHEN row.career IS NULL OR row.career = '' THEN [] ELSE [1] END |
          MERGE (ca:Career {text: row.career})
          MERGE (p)-[:HAS_CAREER]->(ca)
        )
        """
        personality_chunk_cypher = """
        UNWIND $rows AS r
        MERGE (p:Personality {four_letter: r.four_letter})
        MERGE (pa:Passage {
            text: r.text,
            key: r.key,
            chunk_index: r.chunk_index,
            source_file: coalesce(r.source_file, '')
        })
        FOREACH (_ IN CASE WHEN r.embedding IS NULL THEN [] ELSE [1] END | SET pa.embedding = r.embedding)
        MERGE (p)-[:HAS_PASSAGE]->(pa)
        """
        if mode == "csv":
            cypher = mbti_cypher
        elif mode == "json":
            cypher = personality_cypher
        else:
            cypher = personality_chunk_cypher
        with self.driver.session() as session:
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i:i + chunk_size]
                session.run(cypher, {"rows": chunk})
                print("Inserted chunk {} / {}".format(i // chunk_size + 1, math.ceil(len(rows) / chunk_size)))

    def query_neo4j(self, index_name: str = None, embedding: list = None, top_k: int = 5, method: str = "mbti", four_letter: list = None):
        if method == "mbti":
            cypher = """
            MATCH (p:Personality) WHERE p.four_letter IN $mbti_types
            OPTIONAL MATCH (p)-[:HAS_OVERVIEW]->(o:Overview)
            OPTIONAL MATCH (p)-[:HAS_STRENGTH]->(s:Strength)
            OPTIONAL MATCH (p)-[:HAS_WEAKNESS]->(w:Weakness)
            OPTIONAL MATCH (p)-[:HAS_RELATIONSHIP]->(r:Relationship)
            OPTIONAL MATCH (p)-[:HAS_CAREER]->(ca:Career)
            RETURN
                p.four_letter AS type,
                collect(DISTINCT o.text) AS overview,
                collect(DISTINCT s.text) AS strengths,
                collect(DISTINCT w.text) AS weaknesses,
                collect(DISTINCT r.text) AS relationships,
                collect(DISTINCT ca.text) AS careers
            """
            return self.run_cypher(cypher, {"mbti_types": four_letter})
        elif method == "mbti_embedding":
            # cypher = """
            # CALL db.index.vector.queryNodes('{}', $topK, $embedding)
            # YIELD node, score
            # RETURN node, score
            # ORDER BY score DESC
            # """.format(index_name)
            cypher = """
            CALL db.index.vector.queryNodes('{}', $topK, $embedding)
            YIELD node, score
            RETURN node.text AS text, node.source_file AS source, node.key AS key, score
            ORDER BY score DESC
            """.format(index_name)
            return self.run_cypher(cypher, {"embedding": embedding, "topK": top_k})
        elif method == "entity_embedding":
            if index_name == "name_embedding_index":
                cypher = """
                CALL db.index.vector.queryNodes('{}', $topK, $embedding)
                YIELD node AS n, score
                MATCH (n)-[:HAS_TYPE]->(p:Personality)
                OPTIONAL MATCH (p)-[:HAS_OVERVIEW]->(o:Overview)
                OPTIONAL MATCH (p)-[:HAS_STRENGTH]->(s:Strength)
                OPTIONAL MATCH (p)-[:HAS_WEAKNESS]->(w:Weakness)
                OPTIONAL MATCH (p)-[:HAS_RELATIONSHIP]->(r:Relationship)
                OPTIONAL MATCH (p)-[:HAS_CAREER]->(c:Career)
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(sc:Subcategory)<-[:HAS_SUBCATEGORY]-(c:Category)
                RETURN n.name AS name, score,
                   p.four_letter AS personality,
                   collect(DISTINCT sc.name) AS subcategories,
                   collect(DISTINCT c.name) AS categories,
                   collect(DISTINCT o.text) AS overview,
                   collect(DISTINCT s.text) AS strengths,
                   collect(DISTINCT w.text) AS weaknesses,
                   collect(DISTINCT r.text) AS relationships,
                   collect(DISTINCT c.text) AS careers
                ORDER BY score DESC
                """.format(index_name)
            elif index_name == "category_embedding_index":
                cypher = """
                CALL db.index.vector.queryNodes('{}', $topK, $embedding)
                YIELD node AS cat, score
                OPTIONAL MATCH (cat)-[:HAS_SUBCATEGORY]->(sc:Subcategory)<-[:BELONGS_TO]-(p:Personality)
                OPTIONAL MATCH (p)<-[:HAS_TYPE]-(n:Name)
                RETURN cat.name AS category, score,
                       collect(DISTINCT sc.name)        AS subcategories,
                       collect(DISTINCT p.four_letter)  AS personalities,
                       collect(DISTINCT n.name)         AS names
                ORDER BY score DESC
                """.format(index_name)
            elif index_name == "subcategory_embedding_index":
                cypher = """
                CALL db.index.vector.queryNodes('{}', $topK, $embedding)
                YIELD node AS sc, score
                OPTIONAL MATCH (sc)<-[:HAS_SUBCATEGORY]-(cat:Category)
                OPTIONAL MATCH (p:Personality)-[:BELONGS_TO]->(sc)
                OPTIONAL MATCH (p)<-[:HAS_TYPE]-(n:Name)
                RETURN sc.name AS subcategory, score,
                       collect(DISTINCT cat.name)       AS categories,
                       collect(DISTINCT p.four_letter)  AS personalities,
                       collect(DISTINCT n.name)         AS names
                ORDER BY score DESC
                """.format(index_name)
            else:
                raise ValueError("Unknown index: {}".format(index_name))
            return self.run_cypher(cypher, {"embedding": embedding, "topK": top_k})

    def run_cypher(self, cypher: str, params: dict = None):
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]