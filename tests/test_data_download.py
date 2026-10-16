import json
from pathlib import Path

import pytest

from dplm_family_lab import data
from dplm_family_lab.common import read_fasta


def test_fetch_pagination_restart_and_provenance(tmp_path, monkeypatch):
    class Response:
        def __init__(self, content, next_url=None):
            self.content = content
            self.links = {"next": {"url": next_url}} if next_url else {}
            self.headers = {"X-UniProt-Release": "test_release", "X-UniProt-Release-Date": "2026-10-16",
                            "X-Total-Results": "2", "Date": "test-http-date"}

        def raise_for_status(self):
            pass

    class Session:
        def __init__(self):
            self.headers, self.calls, self.fail_once = {}, [], True

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, url, **kwargs):
            self.calls.append(url)
            if "cursor=second" in url:
                if self.fail_once:
                    self.fail_once = False
                    raise ConnectionError("interrupted")
                return Response(b"Entry\tLength\tOrganism (ID)\tReviewed\tSequence\nB\t4\t2\tunreviewed\tCCCC\n")
            return Response(b"\xef\xbb\xbfEntry\tLength\tOrganism (ID)\tReviewed\tSequence\nA\t4\t1\treviewed\tAAAA\n",
                            "https://rest.uniprot.org/uniprotkb/search?cursor=second")

    session = Session()
    monkeypatch.setattr(data.requests, "Session", lambda: session)
    hmm = b"HMMER3/f\nNAME TEST\nACC PF00194.27\nLENG 4\nGA 27 27;\nHMM test\n//\n"
    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: Response(hmm))
    config = {"data": {"raw_dir": str(tmp_path / "raw"), "query": "test", "family": "PF00194"},
              "hmm": {"path": str(tmp_path / "raw/PF00194.hmm"), "url": "https://example.test/model"}}
    with pytest.raises(ConnectionError):
        data.fetch(config)
    result = data.fetch(config)
    assert len([u for u in session.calls if "cursor=second" not in u]) == 1
    assert [r.id for r in read_fasta(result["fasta_path"])] == ["A", "B"]
    provenance = json.loads(Path(result["provenance_path"]).read_text())
    assert provenance["records"] == 2
    assert provenance["uniprot_release"] == "test_release"
    assert provenance["uniprot_release_date"] == "2026-10-16"
    assert len(provenance["pages"]) == 2
    assert all(p["sha256"] and p["retrieved_at"] for p in provenance["pages"])
    assert data.fetch(config)["records"] == 2
    Path(result["fasta_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        data.fetch(config)
