# src/graphraggle/seed_transformer.py
import csv
import io
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD
from slugify import slugify

GRGGL = Namespace("http://grggl.org/")
SCHEMA = Namespace("http://schema.org/")


def csv_to_rdf(csv_path: str | Path) -> Graph:
    """Convert a CSV of seed book data into an RDF graph."""
    path = Path(csv_path)
    con = duckdb.connect()
    df1 = con.execute(f"SELECT * FROM read_csv_auto('{path.as_posix()}')").df()

    graph = Graph()
    graph.namespace_manager.bind("schema", SCHEMA, override=True)
    graph.namespace_manager.bind("grggl", GRGGL, override=True)

    for _, row in df1.iterrows():
        title = str(row["title"]).strip()
        book_uri = URIRef(f"http://grggl.org/book/{slugify(title)}")
        graph.add((book_uri, RDF.type, SCHEMA.Book))
        graph.add((book_uri, SCHEMA.name, Literal(title, datatype=XSD.string)))

        author_name = str(row["author"]).strip()
        author_uri = URIRef(f"http://grggl.org/person/{slugify(author_name)}")
        graph.add((book_uri, SCHEMA.author, author_uri))
        graph.add((author_uri, SCHEMA.name, Literal(author_name, datatype=XSD.string)))

        if "year" in row and row["year"] and not pd.isna(row["year"]):
            graph.add(
                (
                    book_uri,
                    SCHEMA.datePublished,
                    Literal(str(int(row["year"])), datatype=XSD.gYear),
                )
            )

        if "summary" in row and row["summary"] and not pd.isna(row["summary"]):
            graph.add(
                (
                    book_uri,
                    SCHEMA.description,
                    Literal(str(row["summary"]).strip(), datatype=XSD.string),
                )
            )

        if row.get("recognition"):
            for rec_uri in _parse_recognitions(str(row["recognition"])):
                graph.add((book_uri, GRGGL.hasRecognition, rec_uri))

    return graph


def rdf_to_csv(rdf_path: Path, output_path: Path | None = None) -> str | None:
    """Extract structured book data from an RDF graph and return CSV as string."""
    path = Path(rdf_path)
    graph = Graph()
    graph.parse(path)

    rows = []

    for book_uri in graph.subjects(RDF.type, SCHEMA.Book):
        title = graph.value(book_uri, SCHEMA.name)
        author_uri = graph.value(book_uri, SCHEMA.author)
        author_name = None

        if isinstance(author_uri, URIRef):
            # Look up author's name if it's a URI
            author_name = graph.value(author_uri, SCHEMA.name)
        else:
            author_name = author_uri  # If literal

        year = graph.value(book_uri, SCHEMA.datePublished)
        summary = graph.value(book_uri, SCHEMA.description)
        recognitions = _serialize_recognitions(
            [
                uri
                for uri in graph.objects(book_uri, GRGGL.hasRecognition)
                if isinstance(uri, URIRef)
            ]
        )

        rows.append(
            {
                "title": str(title) if title else "",
                "author": str(author_name) if author_name else "",
                "year": str(year) if year else "",
                "summary": str(summary) if summary else "",
                "recognition": recognitions,
            }
        )

    if not rows:
        return None

    # Convert to CSV string
    fieldnames = ["title", "author", "year", "summary", "recognition"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    csv_string = output.getvalue()

    # Optionally write to file via DuckDB
    if output_path:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(csv_string)

        con = duckdb.connect()
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{tmp_path.as_posix()}')) \
                TO '{Path(output_path).as_posix()}' (FORMAT CSV, HEADER)"
        )
        tmp_path.unlink()

    return csv_string


def _parse_recognitions(field: str) -> list[URIRef]:
    """Convert semicolon- or comma-separated recognitions to URIRefs."""
    items = [item.strip() for item in re.split(r"[;,]", field)]
    cleaned = list({item for item in items if item})
    return [_build_recognition_uri(item) for item in cleaned]


def _serialize_recognitions(uris: Iterable[URIRef]) -> str:
    """Flatten URIs to readable recognition names separated by semicolons."""
    return "; ".join(uri.rsplit("/", 1)[-1].replace("_", " ").title() for uri in uris)


def _build_recognition_uri(name: str) -> URIRef:
    """Generate a URIRef from a normalized recognition name."""
    return URIRef(f"http://grggl.org/recognition/{slugify(name)}")
