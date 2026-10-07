"""
Tests for the NVIDIA NIM client in src/downloader.py (MSA search + OpenFold3).

No network and no credits: requests.post / requests.get / time.sleep are
replaced by fakes that record what would have been sent.
"""

import json
import os

import pytest
import requests
from requests.structures import CaseInsensitiveDict

import src.downloader as dl

SEQ = "MKTAYIAKQR"
A3M = f">101\n{SEQ}\n>UniRef100_X\nMKTAYLAKQR\n>UniRef100_Y\nMRTAYIAK-R\n"
ENV_A3M = f">101\n{SEQ}\n>env_1\nMKSAYIAKQR\n"
MERGED_A3M = A3M + ENV_A3M  # the service's extra "colabfold" alignment
CIF = "data_{name}\n_entry.id {name}\n"


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = CaseInsensitiveDict(headers or {})
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


@pytest.fixture
def nim(monkeypatch):
    """Queue fake answers; record every request and every sleep."""
    state = {"answers": [], "requests": [], "sleeps": []}

    def answer():
        item = state["answers"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def fake_post(url, json=None, headers=None, timeout=None):
        state["requests"].append(("POST", url, json))
        return answer()

    def fake_get(url, headers=None, timeout=None):
        state["requests"].append(("GET", url, None))
        return answer()

    monkeypatch.setattr(dl.requests, "post", fake_post)
    monkeypatch.setattr(dl.requests, "get", fake_get)
    monkeypatch.setattr(dl.time, "sleep", lambda s: state["sleeps"].append(s))
    monkeypatch.setenv("NIM_API_KEY", "nvapi-test")
    return state


def _msa_answer():
    return FakeResponse(
        200,
        {
            "alignments": {
                "Uniref30_2302": {"a3m": {"alignment": A3M, "format": "a3m"}},
                "colabfold_envdb_202108": {
                    "a3m": {"alignment": ENV_A3M, "format": "a3m"}
                },
                "colabfold": {"a3m": {"alignment": MERGED_A3M, "format": "a3m"}},
            }
        },
    )


def _of3_answer(confidences):
    samples = [
        {"structure": CIF.format(name=f"s{k}"), "confidence_score": c}
        for k, c in enumerate(confidences)
    ]
    return FakeResponse(200, {"outputs": [{"structures_with_scores": samples}]})


# --- MSA search -------------------------------------------------------------


def test_fetch_msa_sends_colabfold_search_and_caches(nim, tmp_path):
    nim["answers"].append(_msa_answer())
    msa = dl.fetch_msa("P1", SEQ, save_dir=str(tmp_path))

    method, url, payload = nim["requests"][0]
    assert (method, url) == ("POST", dl._NIM_MSA_URL)
    assert payload["sequence"] == SEQ
    assert payload["search_type"] == "colabfold"
    assert payload["databases"] == list(dl.MSA_DATABASES)
    assert msa == {"Uniref30_2302": A3M, "colabfold_envdb_202108": ENV_A3M}

    info = json.loads((tmp_path / "P1" / "msa_search.json").read_text())
    assert info["n_sequences"] == {
        "Uniref30_2302": 3,
        "colabfold_envdb_202108": 2,
        "colabfold": 5,
    }

    # second call: read from disk, no request
    assert dl.fetch_msa("P1", SEQ, save_dir=str(tmp_path)) == msa
    assert len(nim["requests"]) == 1


def test_merged_alignment_is_saved_but_not_used(nim, tmp_path):
    # the service also returns "colabfold" = the requested alignments merged;
    # passing it as well would give OpenFold3 every sequence twice
    nim["answers"].append(_msa_answer())
    msa = dl.fetch_msa("P1", SEQ, save_dir=str(tmp_path))
    assert "colabfold" not in msa
    assert (tmp_path / "P1" / "colabfold.a3m").read_text() == MERGED_A3M
    assert "colabfold" not in dl.fetch_msa("P1", SEQ, save_dir=str(tmp_path))


def test_fetch_msa_rejects_nonstandard_letters(nim, tmp_path):
    with pytest.raises(ValueError, match="standard amino acids"):
        dl.fetch_msa("P1", "MKUX", save_dir=str(tmp_path))
    assert nim["requests"] == []


def test_fetch_msa_searches_again_if_settings_or_files_change(nim, tmp_path):
    nim["answers"] += [_msa_answer(), _msa_answer(), _msa_answer()]
    dl.fetch_msa("P1", SEQ, save_dir=str(tmp_path))
    dl.fetch_msa("P1", SEQ, save_dir=str(tmp_path), max_msa_sequences=2000)
    assert len(nim["requests"]) == 2  # other settings: not the cached MSA
    (tmp_path / "P1" / "colabfold_envdb_202108.a3m").unlink()
    dl.fetch_msa("P1", SEQ, save_dir=str(tmp_path), max_msa_sequences=2000)
    assert len(nim["requests"]) == 3  # a file is missing: search again


# --- OpenFold3 ----------------------------------------------------------------


def test_predict_sends_msa_and_keeps_top_ranked_sample(nim, tmp_path):
    nim["answers"].append(_of3_answer([0.50, 0.91, 0.70]))
    msa = {"Uniref30_2302": A3M, "colabfold_envdb_202108": ENV_A3M}
    path = dl.predict_openfold3("P1", SEQ, str(tmp_path), msa=msa)

    _, url, payload = nim["requests"][0]
    assert url == dl._NIM_OF3_URL
    molecule = payload["inputs"][0]["molecules"][0]
    assert molecule["sequence"] == SEQ
    assert molecule["msa"]["Uniref30_2302"]["a3m"] == {
        "alignment": A3M,
        "format": "a3m",
    }
    assert payload["inputs"][0]["diffusion_samples"] == 5

    assert open(path).read() == CIF.format(name="s1")  # highest confidence
    assert len(os.listdir(tmp_path / "samples")) == 3
    prov = json.loads((tmp_path / "OF3_P1_provenance.json").read_text())
    assert prov["selected_sample"] == 1
    assert prov["msa_n_sequences"] == {"Uniref30_2302": 3, "colabfold_envdb_202108": 2}
    assert [s["confidence_score"] for s in prov["sample_scores"]] == [0.50, 0.91, 0.70]


def test_predict_needs_an_msa(nim, tmp_path):
    with pytest.raises(TypeError):
        dl.predict_openfold3("P1", SEQ, str(tmp_path))  # msa is required
    with pytest.raises(ValueError, match="no MSA"):
        dl.predict_openfold3("P1", SEQ, str(tmp_path), msa={})
    assert nim["requests"] == []


def test_predict_rejects_msa_of_another_sequence(nim, tmp_path):
    with pytest.raises(ValueError, match="not the query"):
        dl.predict_openfold3("P1", SEQ[:-1], str(tmp_path), msa={"u": A3M})
    assert nim["requests"] == []


def test_empty_structure_is_rejected(nim, tmp_path):
    answer = _of3_answer([0.9])
    answer.json()["outputs"][0]["structures_with_scores"][0]["structure"] = ""
    nim["answers"].append(answer)
    with pytest.raises(ValueError, match="not valid mmCIF"):
        dl.predict_openfold3("P1", SEQ, str(tmp_path), msa={"u": A3M})
    assert not (tmp_path / "OF3_P1.cif").exists()


def test_model_is_written_last(nim, tmp_path, monkeypatch):
    nim["answers"].append(_of3_answer([0.9]))

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dl.json, "dump", broken_dump)
    with pytest.raises(OSError):
        dl.predict_openfold3("P1", SEQ, str(tmp_path), msa={"u": A3M})
    assert not (tmp_path / "OF3_P1.cif").exists()  # no model without provenance


def test_model_without_provenance_is_refused(nim, tmp_path):
    (tmp_path / "OF3_P1.cif").write_text(CIF.format(name="other"))
    with pytest.raises(RuntimeError, match="without"):
        dl.predict_openfold3("P1", SEQ, str(tmp_path), msa={"u": A3M})


def test_msa_free_file_is_refused(nim, tmp_path):
    (tmp_path / "OF3_P1.cif").write_text(CIF.format(name="other"))
    (tmp_path / "OF3_P1_provenance.json").write_text(
        json.dumps({"msa_type": "single_sequence_query_only"})
    )
    with pytest.raises(RuntimeError, match="without an MSA"):
        dl.predict_openfold3("P1", SEQ, str(tmp_path), msa={"u": A3M})


# --- retries and long jobs ------------------------------------------------------


def test_rate_limit_is_retried(nim):
    nim["answers"] += [FakeResponse(429, headers={"Retry-After": "7"}), _msa_answer()]
    result = dl._post_nim(dl._NIM_MSA_URL, {}, "key").json()
    assert "alignments" in result
    assert nim["sleeps"] == [7.0]


def test_rate_limit_gives_up_after_max_retries(nim):
    nim["answers"] += [FakeResponse(429) for _ in range(3)]
    with pytest.raises(requests.HTTPError, match="429"):
        dl._post_nim(dl._NIM_MSA_URL, {}, "key", max_retries=2)
    assert nim["sleeps"] == [10.0, 20.0]


def test_running_job_is_polled(nim):
    nim["answers"] += [
        FakeResponse(202, headers={"NVCF-REQID": "req-1"}),
        FakeResponse(202, headers={"NVCF-REQID": "req-1"}),
        _of3_answer([0.8]),
    ]
    result = dl._post_nim(dl._NIM_OF3_URL, {}, "key").json()
    assert result["outputs"]
    assert [r[:2] for r in nim["requests"]] == [
        ("POST", dl._NIM_OF3_URL),
        ("GET", f"{dl._NIM_STATUS_URL}/req-1"),
        ("GET", f"{dl._NIM_STATUS_URL}/req-1"),
    ]


def test_error_while_polling_does_not_submit_the_job_again(nim):
    nim["answers"] += [
        FakeResponse(202, headers={"NVCF-REQID": "req-1"}),
        FakeResponse(503),
        _of3_answer([0.8]),
    ]
    dl._post_nim(dl._NIM_OF3_URL, {}, "key")
    assert [r[0] for r in nim["requests"]] == ["POST", "GET", "GET"]


def test_running_job_without_request_id_is_an_error(nim):
    nim["answers"].append(FakeResponse(202))
    with pytest.raises(requests.HTTPError, match="request id"):
        dl._post_nim(dl._NIM_OF3_URL, {}, "key")


def test_time_out_on_submission_is_retried(nim):
    nim["answers"] += [requests.ReadTimeout("read timed out"), _msa_answer()]
    assert "alignments" in dl._post_nim(dl._NIM_MSA_URL, {}, "key").json()
    assert nim["sleeps"] == [10.0]


def test_time_out_while_polling_polls_the_same_job(nim):
    nim["answers"] += [
        FakeResponse(202, headers={"NVCF-REQID": "req-1"}),
        requests.ReadTimeout("read timed out"),
        _of3_answer([0.8]),
    ]
    dl._post_nim(dl._NIM_OF3_URL, {}, "key")
    assert [r[0] for r in nim["requests"]] == ["POST", "GET", "GET"]


def test_invalid_request_is_not_retried(nim):
    nim["answers"].append(FakeResponse(422, {"detail": "bad msa"}))
    with pytest.raises(requests.HTTPError, match="422"):
        dl._post_nim(dl._NIM_OF3_URL, {}, "key")
    assert nim["sleeps"] == []


def test_af2_download_saves_provenance(tmp_path, nim):
    record = {"entryId": "AF-P1-F1", "latestVersion": 4, "modelCreatedDate": "2022-06-01",
              "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P1-F1-model_v4.cif", "uniprotSequence": SEQ}
    cif = FakeResponse(200)
    cif.text = CIF.format(name="AF-P1-F1")
    nim["answers"] += [FakeResponse(200, [record]), cif]
    path = dl.download_alphafold2("P1", str(tmp_path))
    with open(path.replace(".cif", "_provenance.json")) as f:
        prov = json.load(f)
    assert prov["latestVersion"] == 4 and prov["entryId"] == "AF-P1-F1"
    assert prov["cifUrl"].endswith("model_v4.cif") and "uniprotSequence" not in prov
    assert len(prov["cif_sha256"]) == 64
