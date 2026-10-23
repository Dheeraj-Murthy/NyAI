"""
Smoke Test for document file resolution (no database needed)
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with tempfile.TemporaryDirectory() as root:
    os.environ["DOCUMENTS_ROOT"] = root
    import documents  # reads DOCUMENTS_ROOT at import

    (Path(root) / "data" / "statutes").mkdir(parents=True)
    pdf = Path(root) / "data" / "archive" / "case one.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4 test")
    (Path(root) / "data" / "statutes" / "BNS.pdf").write_bytes(b"%PDF-1.4 bns")
    (Path(root) / "data" / "statutes" / "BNSS.pdf").write_bytes(b"%PDF-1.4 bnss")
    (Path(root) / "notes.txt").write_text("not a pdf")

    print("--- Testing document file resolution ---")
    assert documents.relative_source_path(str(pdf)) == "data/archive/case one.pdf"
    assert documents.relative_source_path("/etc/hosts") == "/etc/hosts"  # outside root: kept, never served

    assert documents.resolve_source_file("data/archive/case one.pdf") == pdf.resolve()
    for unsafe in ["../../../../etc/passwd", "/etc/hosts", "notes.txt", "data/archive/missing.pdf", "", None]:
        assert documents.resolve_source_file(unsafe) is None, unsafe

    assert documents.statute_matches_file("BNS", "BNS.pdf")
    assert documents.statute_matches_file("Constitution", "constitution of india.pdf")
    assert not documents.statute_matches_file("Constitution", "BNS.pdf")
    assert documents.statute_matches_file("BNSS", "BNSS.pdf")
    assert not documents.statute_matches_file("BNS", "BNSS.pdf")  # "bns" is inside "bnss"
    assert not documents.statute_matches_file("BNSS", "BNS.pdf")
    assert documents._statute_file_by_name("BNS").name == "BNS.pdf"
    assert documents._statute_file_by_name("BNSS").name == "BNSS.pdf"
    assert documents._statute_file_by_name("Constitution") is None
    print("✓ document file resolution passed")
