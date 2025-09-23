import concurrent.futures
import io
import logging
import requests
from requests.auth import HTTPBasicAuth
from contextlib import contextmanager
from typing import Any, Optional, Tuple
import math

import mysql.connector

from vectordb_bench import config

from ..api import MetricType, VectorDB
from .config import DorisCaseConfig

log = logging.getLogger(__name__)


class Doris(VectorDB):
    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: DorisCaseConfig,
            drop_old: bool = False,
            **kwargs,
    ):
        self.name = "Doris"
        self.db_config = db_config
        self.case_config = db_case_config
        self.dim = dim
        self.search_fn = db_case_config.search_param()["metric_fn"]
        # e.g. l2_distance128, inner_product128
        self.table_name = self.search_fn + str(dim)
        self.conn = None  # To be inited by init()
        self.cursor = None  # To be inited by init()

        if drop_old:
            self._drop_table()
            self._create_table()

    @contextmanager
    def init(self):
        with self._get_connection() as (conn, cursor):
            self.conn = conn
            self.cursor = cursor
            try:
                yield
            finally:
                self.conn = None
                self.cursor = None

    @contextmanager
    def _get_connection(self):
        # Only pass MySQL-compatible keys to mysql-connector
        mysql_keys = {
            "host",
            "port",
            "user",
            "password",
            "database",
        }
        mysql_cfg = {k: v for k, v in self.db_config.items() if k in mysql_keys}
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(
                host=mysql_cfg.get("host"),
                port=mysql_cfg.get("port"),
                user=mysql_cfg.get("user"),
                password=mysql_cfg.get("password"),
                database=mysql_cfg.get("database"),
            )
            # Use prepared cursor to enable server-side prepared statements
            cursor = conn.cursor(prepared=True)
            # Apply session variables (defaults + user-provided)
            default_session = {"parallel_pipeline_task_num": "1"}
            user_session = {}
            try:
                user_session = self.case_config.session_param() if hasattr(self.case_config, "session_param") else {}
            except Exception:
                user_session = {}
            session_vars = {**default_session, **(user_session or {})}

            def _fmt_val(v: any) -> str:
                if v is None:
                    return "NULL"
                vs = str(v)
                # simple numeric check
                try:
                    float(vs)
                    return vs
                except Exception:
                    pass
                # quote string
                return "'" + vs.replace("'", "''") + "'"

            for k, v in session_vars.items():
                try:
                    cursor.execute(f"set {k}={_fmt_val(v)};")
                except Exception as e:
                    log.warning("Failed to set session var %s=%s: %s", k, v, e)
            yield conn, cursor
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                if conn is not None:
                    conn.close()

    def _drop_table(self):
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                conn.commit()
        except Exception as e:
            log.warning("Failed to drop table: %s error: %s", self.table_name, e)
            raise e

    def _create_table(self):
        try:
            index_param = self.case_config.index_param()
            with self._get_connection() as (conn, cursor):
                log.info("Creating table %s with index %s", self.table_name, index_param)
                metric = index_param.get("metric_fn", "l2_distance")

                # Determine buckets by counting alive backends
                def _get_alive_be_count(cn):
                    try:
                        cur = cn.cursor()  # non-prepared cursor for SHOW statements
                        cur.execute("SHOW BACKENDS")
                        rows = cur.fetchall()
                        col_names = getattr(cur, "column_names", None)
                        if not col_names and cur.description:
                            col_names = [d[0] for d in cur.description]
                        alive_idx = None
                        if col_names:
                            for i, n in enumerate(col_names):
                                if str(n).lower() == "alive":
                                    alive_idx = i
                                    break
                        count = 0
                        if alive_idx is None:
                            # Fallback: assume all rows are backends
                            count = len(rows)
                        else:
                            for r in rows:
                                sval = str(r[alive_idx]).strip().lower()
                                if sval in ("true", "1", "yes", "y"):
                                    count += 1
                        cur.close()
                        return max(1, count)
                    except Exception as e:
                        log.warning("SHOW BACKENDS failed, fallback to 1 bucket: %s", e)
                        return 1

                buckets = _get_alive_be_count(conn)
                log.info("Using %d BUCKETS according to alive backends", buckets)

                # Compose index properties
                idx_props = {
                    "index_type": "hnsw",
                    "metric_type": str(metric),
                    "dim": str(self.dim),
                    "ef_construction": str(index_param.get("ef_construction", 40)),
                    "max_degree": str(index_param.get("max_degree", 32)),
                }
                # Merge additional user properties except internal helper keys
                for k, v in index_param.items():
                    if k in {"metric_fn"}:
                        continue
                    if v is None:
                        continue
                    idx_props[str(k)] = str(v)
                idx_props_str = ",\n\t\t\t\t".join([f'"{k}"="{v}"' for k, v in idx_props.items()])

                ddl = f"""
                    CREATE TABLE {self.table_name} (
                        id BIGINT NOT NULL,
                        embedding ARRAY<FLOAT> NOT NULL,
                        INDEX idx_emb (`embedding`) USING ANN PROPERTIES(
                            {idx_props_str}))
                        DUPLICATE KEY(`id`)
                        DISTRIBUTED BY HASH(`id`) BUCKETS {buckets}
                        PROPERTIES (
                            "replication_num" = "1"
                        );
                        """
                log.info("Create table DDL: %s", ddl)
                cursor.execute(ddl)
                conn.commit()
        except Exception as e:
            log.warning("Failed to create table: %s error: %s", self.table_name, e)
            raise e

    def ready_to_load(self) -> bool:
        pass

    def optimize(self, data_size: int | None = None) -> None:
        self.cursor.execute("set parallel_pipeline_task_num=1;")
        log.info("Nothing to do now")

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        if self.case_config.metric_type == MetricType.COSINE:
            log.info("cosine dataset need normalize.")
            return True

        return False

    def _insert_embeddings_serial(
            self,
            embeddings: list[list[float]],
            metadata: list[int],
            offset: int,
            size: int,
    ) -> Exception:
        try:
            buf = io.StringIO()
            for i in range(offset, offset + size):
                if i > offset:
                    buf.write("\n")
                astr = ','.join(map(str, embeddings[i]))
                buf.write(f"{metadata[i]},'[{astr}]'")
            v = buf.getvalue()
            self.send_request(v)

        except Exception as e:
            log.warning("Failed to insert data into table: %s", e)
            raise e

    def send_request(self, body: str):
        # Print self.db_config here
        log.debug(f"DB Config: {self.db_config}")
        # Determine scheme and HTTP port
        ssl_enabled = bool(self.db_config.get("ssl_verify_cert") or self.db_config.get("ssl_verify_identity"))
        scheme = "https" if ssl_enabled else "http"
        default_http_port = 8040 if ssl_enabled else 8030
        http_port = int(self.db_config.get("http_port") or default_http_port)
        # Build stream load URL, use class table name
        url = (
            f"{scheme}://{self.db_config.get('host')}:{http_port}/api/"
            f"{self.db_config.get('database')}/{self.table_name}/_stream_load"
        )
        log.info(f"Stream load to {url}, length {len(body)/1024/1024} MB")
        # Build Basic Auth from configured user/password
        user = self.db_config.get("user", "root")
        pwd = self.db_config.get("password", "")
        auth = HTTPBasicAuth(user, pwd)
        for _ in range(3):
            headers = {
                "Content-Type": "text/csv",
                "Expect": "100-continue",
                "format": "csv",
                "column_separator": ",",
                "columns": "id,embedding",
                "enclose": "'",
                "trim_double_quotes": "false",
            }

            try:
                session = requests.sessions.Session()
                session.should_strip_auth = lambda old_url, new_url: False  # Don't strip auth
                response = session.request("PUT", url, data=bytearray(body.encode("utf-8")), headers=headers, timeout=36000, auth=auth)
                response.raise_for_status()
                resbody = response.json()
                if resbody.get("Status") != "Success":
                    logging.error(f"Response message: {resbody}")
                    if resbody.get("Status") != "Publish Timeout":
                        raise Exception(f"Response message: {resbody}")
                return

            except requests.exceptions.HTTPError:
                if response.status_code == 307:
                    url = response.headers.get("Location", url)
                    logging.info(f"Redirect to {url}")
                    continue
                try:
                    resbody = response.json()
                except Exception:
                    resbody = response.text if response is not None else ""
                logging.error(
                    f"Response code is not 200, code: {getattr(response, 'status_code', 'unknown')}, response: {resbody}"
                )
                raise
            except Exception:
                logging.exception("Stream load request failed")

        raise Exception("Redirect too much or request failed repeatedly")

    def insert_embeddings(
            self,
            embeddings: list[list[float]],
            metadata: list[int],
            **kwargs: Any,
    ) -> Tuple[int, Optional[Exception]]:
        """Insert embeddings concurrently.

        - Batch by config.NUM_PER_BATCH to avoid exceeding MAX_ALLOWED_PACKET (default 64MB).
        - Number of workers equals number of batches: ceil(num_rows / batch_size).
        """
        batch_size = (
            int(self.case_config.stream_load_rows_per_batch)
            if getattr(self.case_config, "stream_load_rows_per_batch", None)
            else config.NUM_PER_BATCH
        )
        workers = max(1, math.ceil(len(embeddings) / batch_size))
        log.info(
            f"Insert {len(embeddings)} embeddings with batch size {batch_size}, workers {workers}"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures: list[concurrent.futures.Future] = []
            for i in range(0, len(embeddings), batch_size):
                offset = i
                size = min(batch_size, len(embeddings) - i)
                future = executor.submit(
                    self._insert_embeddings_serial, embeddings, metadata, offset, size
                )
                futures.append(future)

            done, pending = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            )
            executor.shutdown(wait=False)
            for future in done:
                future.result()
            for future in pending:
                future.cancel()

        return len(metadata), None

    def search_embedding(
            self,
            query: list[float],
            k: int = 100,
            filters: dict | None = None,
            timeout: int | None = None,
            **kwargs: Any,
    ) -> list[int]:
        sql, params = self.search_sql_prepared(self.search_fn, query, k, filters, timeout)
        self.cursor.execute(sql, params)
        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]

    def search_embedding_range(
            self,
            query: list[float],
            k: int = 100,
            filters: dict | None = None,
            distance: float | None = None,
            timeout: int | None = None,
            **kwargs: Any,
    ) -> list[int]:
        sql, params = self.search_range_sql_prepared(self.search_fn, query, k, filters, distance)
        # log.info("Executing search SQL: %s", sql)
        self.cursor.execute(sql, params)
        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]

    def search_embedding_compound(
            self,
            query: list[float],
            k: int = 100,
            filters: dict | None = None,
            distance: float | None = None,
            timeout: int | None = None,
            **kwargs: Any,
    ) -> list[int]:
        sql, params = self.search_compound_sql_prepared(self.search_fn, query, k, filters, distance)
        # log.info("Executing search SQL: %s", sql)
        self.cursor.execute(sql, params)
        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]

    def search_distance(self,
                        query: list[float],
                        id: int | None = None):
        sql = f"SELECT {self.search_fn}(embedding, %s) FROM {self.table_name} WHERE id = %s"
        self.cursor.execute(sql, (str(query), id))
        result = self.cursor.fetchall()
        return [float(i[0]) for i in result]

    def search_embedding_exact(
            self,
            query: list[float],
            k: int = 100,
            filters: dict | None = None,
            timeout: int | None = None,
            **kwargs: Any,
    ) -> list[int]:
        # 删除 approximate 后缀，这时返回的值认为是准确的，用来计算自己的召回率
        metric_type = self.search_fn
        if metric_type.endswith("_approximate"):
            metric_type = metric_type.replace("_approximate", "")

        sql, params = self.search_sql_prepared(metric_type, query, k, filters, timeout)
        self.cursor.execute(sql, params)
        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]

    def search_sql_prepared(self,
                   metric_type: str,
                   query: list[float],
                   k: int = 100,
                   filters: dict | None = None,
                   timeout: int | None = None,
                   ):
        params = []
        
        if filters is not None:
            if 'id' in filters:
                if self.search_fn.startswith("inner_product"):
                    sql = f"""
                        SELECT id FROM {self.table_name}
                        WHERE id >= %s
                        ORDER BY {metric_type}(embedding, %s) DESC LIMIT {k}
                        """
                    params = [filters['id'], str(query)]
                    return sql, params

                sql = f"""
                    SELECT id FROM {self.table_name}
                    WHERE id < %s
                    ORDER BY {metric_type}(embedding, %s) LIMIT {k}
                    """
                params = [filters['id'], str(query)]
                return sql, params

            raise ValueError("filter is not None but id is not in filter")

        if self.search_fn.startswith("inner_product"):
            sql = f"""
            SELECT id FROM {self.table_name}
            ORDER BY {metric_type}(embedding, CAST(%s AS ARRAY<FLOAT>)) DESC LIMIT {k}
            """
            params = [str(query),]
            return sql, params
        else:
            sql = f"""
                SELECT id FROM {self.table_name}
                ORDER BY {metric_type}(embedding, CAST(%s AS ARRAY<FLOAT>)) LIMIT {k}
                """
            params = [str(query),]
            return sql, params

    def search_range_sql_prepared(self,
                   metric_type: str,
                   query: list[float],
                   k: int = 100,
                   filters: dict | None = None,
                   distance: float | None = None,
                   ):
        params = []
        
        if self.search_fn.startswith("inner_product"):
            adjusted_distance = distance - 0.000001
            sql = f"""
                SELECT id FROM {self.table_name}
                WHERE {metric_type}(embedding, CAST(%s AS ARRAY<FLOAT>)) >= {adjusted_distance}
                """
            params = [str(query),]
            return sql, params
        else:
            adjusted_distance = distance + 0.000001
            sql = f"""
                SELECT id FROM {self.table_name}
                WHERE {metric_type}(embedding, CAST(%s AS ARRAY<FLOAT>)) < {adjusted_distance}
                """
            params = [str(query),]
            return sql, params

    def search_compound_sql_prepared(self,
                   metric_type: str,
                   query: list[float],
                   k: int = 100,
                   filters: dict | None = None,
                   distance: float | None = None,
                   ):
        params = []
        
        if self.search_fn.startswith("inner_product"):
            adjusted_distance = distance - 0.000001
            sql = f"""
                SELECT id FROM {self.table_name}
                WHERE {metric_type}(embedding, CAST(%s as ARRAY<FLOAT>)) >= %s
                ORDER BY {metric_type}(embedding, CAST(%s as ARRAY<FLOAT>)) DESC
                LIMIT {k}
                """
            params = [str(query), adjusted_distance, str(query)]
            return sql, params

        adjusted_distance = distance + 0.000001
        sql = f"""
            SELECT id FROM {self.table_name}
            WHERE {metric_type}(embedding, CAST(%s AS ARRAY<FLOAT>)) < %s 
            ORDER BY {metric_type}(embedding, CAST(%s AS ARRAY<FLOAT>)) 
            LIMIT {k}
            """
        params = [str(query), adjusted_distance, str(query)]
        return sql, params