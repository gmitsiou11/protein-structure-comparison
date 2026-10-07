"""
Unit tests for src/dataset.py (offline: tiny tables in the real file formats).

Barnase (P00648) is the worked example: UniProt length 157, signal peptide
1–34, propeptide 35–47, mature chain 48–157 (110 residues).
"""

import gzip

import pandas as pd
import pytest

import src.dataset as ds

SIFTS = """# 2026/09/20 - 12:00 | PDB: 39.26 | UniProt: 2026.04
PDB,CHAIN,SP_PRIMARY,RES_BEG,RES_END,PDB_BEG,PDB_END,SP_BEG,SP_END
1bnr,A,P00648,1,110,1,110,48,157
6pqk,A,P00648,3,110,3,110,50,157
7abc,A,P00648,1,60,1,60,48,107
7abc,B,P00648,1,110,1,110,48,157
8emx,C,P00648,5,105,5,105,52,152
1e10,A,P11111,1,90,1,90,1,90
1e10,A,P22222,91,150,91,150,1,60
2iso,A,P44444-2,1,80,1,80,1,80
3rev,A,P55555,1,80,1,80,90,10
9nmr,NA,P33333,1,120,1,120,1,120
9emm,1,P33333,1,118,1,118,2,119
"""
ENTRY_TYPE = (
    "1bnr\tprot\tNMR\n6pqk\tprot\tdiffraction\n7abc\tprot\tNMR\n8emx\tprot-nuc\tEM\n"
    "1e10\tprot\tdiffraction\n9nmr\tprot\tNMR\n9emm\tprot-nuc\tEM\n100d\tnuc\tdiffraction\n"
)
RESOLU = (
    " PROTEIN DATA BANK LOCAL INDEX DATA -- RESOLUTION\n\nIDCODE\t\tRESOLUTION\n------\t\t----------\n"
    "1BNR\t;\t-1.00\n6PQK\t;\t1.20\n7ABC\t;\t-1.00\n8EMX\t;\t3.10\n1E10\t;\t2.00\n9NMR\t;\t\n9EMM\t;\t2.80\n"
)


@pytest.fixture
def tables(tmp_path):
    with gzip.open(tmp_path / "seg.csv.gz", "wt") as f:
        f.write(SIFTS)
    (tmp_path / "type.txt").write_text(ENTRY_TYPE)
    (tmp_path / "resolu.idx").write_text(RESOLU)
    segments = ds.load_segments(str(tmp_path / "seg.csv.gz"))
    chains = ds.chain_table(
        segments,
        ds.load_methods(str(tmp_path / "type.txt")),
        ds.load_resolution(str(tmp_path / "resolu.idx")),
    )
    return tmp_path, segments, chains


def _facts():
    facts = pd.DataFrame(
        {
            "uniprot": ["P00648", "P33333"],
            "length": [157, 120],
            "protein_name": ["Ribonuclease (EC 3.1.27.-) (Barnase)", "Test protein"],
            "organism": ["Bacillus amyloliquefaciens", "Escherichia coli"],
            "chain_field": ['CHAIN 48..157; /note="Ribonuclease"', ""],
        }
    )
    mature = [ds.parse_mature(c, n) for c, n in zip(facts.chain_field, facts.length)]
    facts[["mature_start", "mature_end", "n_chains"]] = pd.DataFrame(mature)
    facts["mature_length"] = facts["mature_end"] - facts["mature_start"] + 1
    return facts


def test_sifts_version(tables):
    tmp_path, _, _ = tables
    assert ds.sifts_version(str(tmp_path / "seg.csv.gz")).startswith("2026/09/20")


def test_load_segments_drops_chimeras_isoforms_and_reversed_rows(tables):
    _, segments, _ = tables
    assert "1e10" not in set(segments["pdb"])  # one chain, two proteins
    assert "P44444-2" not in set(segments["uniprot"])  # isoform
    assert "3rev" not in set(segments["pdb"])  # SP_END < SP_BEG
    assert {"NA", "1"} <= set(segments["chain"])  # chain IDs stay text


def test_methods_and_resolution(tables):
    _, _, chains = tables
    methods = dict(zip(chains["pdb"], chains["method"]))
    assert (
        methods["6pqk"] == "X-ray"
        and methods["8emx"] == "Cryo-EM"
        and methods["1bnr"] == "NMR"
    )
    assert (
        chains.loc[chains["pdb"] == "1bnr", "resolution"].isna().all()
    )  # NMR: no resolution


def test_quality_limits():
    chains = pd.DataFrame(
        {
            "pdb": ["a", "b", "c", "d"],
            "chain": ["A"] * 4,
            "uniprot": ["P1"] * 4,
            "method": ["X-ray", "X-ray", "Cryo-EM", "Cryo-EM"],
            "resolution": [2.4, 2.6, 3.9, 4.1],
        }
    )
    segments = chains[["pdb", "chain", "uniprot"]].assign(sp_beg=1, sp_end=10)
    kept = ds.chain_table(
        segments, chains[["pdb", "method"]], chains[["pdb", "resolution"]]
    )
    assert set(kept["pdb"]) == {"a", "c"}


@pytest.mark.parametrize(
    "field, length, expected",
    [
        ('CHAIN 48..157; /note="Ribonuclease"', 157, (48, 157, 1)),
        ("", 120, (1, 120, 0)),
        ("CHAIN 1..99; /note=a; CHAIN 100..300; /note=b", 500, (1, 99, 2)),
        ("CHAIN ?..157; /note=x", 157, (1, 157, 1)),
    ],
)
def test_parse_mature(field, length, expected):
    assert ds.parse_mature(field, length) == expected


def test_coverage_is_measured_on_the_mature_chain(tables):
    _, segments, chains = tables
    cov = ds.add_coverage(chains, segments, _facts())
    xray = cov[(cov["pdb"] == "6pqk")].iloc[0]
    assert xray["coverage"] == pytest.approx(108 / 110)  # 98 %, not 108/157 = 69 %
    partial = cov[(cov["pdb"] == "7abc") & (cov["chain"] == "A")].iloc[0]
    assert partial["coverage"] == pytest.approx(60 / 110)


def test_filters_counts_and_manifest(tables, tmp_path):
    _, segments, chains = tables
    facts = _facts()
    cov = ds.add_coverage(
        chains[chains["uniprot"].isin(ds.proteins_with(chains))], segments, facts
    )
    both = lambda d: d[d["uniprot"].isin(ds.proteins_with(d))]
    kept, counts = ds.filter_with_counts(
        cov,
        [
            ("coverage >= 80%", lambda d: both(d[d["coverage"] >= 0.8])),
            ("only P00648", lambda d: d[d["uniprot"] == "P00648"]),
        ],
        start_label="candidates",
    )
    assert counts["proteins"].tolist() == [2, 2, 1]
    reps = ds.pick_representatives(kept)
    row = reps.iloc[0]
    assert (row["pdb_nmr"], row["chain_nmr"]) == (
        "1BNR",
        "A",
    )  # full coverage, alphabetical
    assert (row["pdb_cryoem"], row["chain_cryoem"]) == ("8EMX", "C")
    ds.write_manifest(reps, facts, path=str(tmp_path / "proteins.csv"))
    back = pd.read_csv(tmp_path / "proteins.csv", dtype=str, keep_default_na=False)
    assert back.loc[0, "protein_name"] == "Ribonuclease"
    assert back.loc[0, "mature_start"] == "48"


def test_shared_protein_names_are_made_unique(tmp_path):
    facts = pd.DataFrame(
        {
            "uniprot": ["P1", "P2", "P3"],
            "protein_name": [
                "Large ribosomal subunit protein uL2",
                "Large ribosomal subunit protein uL2",
                "Ubiquitin",
            ],
            "organism": ["Escherichia coli", "Thermus thermophilus", "Homo sapiens"],
            "mature_start": [1, 1, 1],
            "mature_end": [100, 100, 76],
        }
    )
    reps = pd.DataFrame({"uniprot_id": ["P1", "P2", "P3"]})
    for col in ["xray", "nmr", "cryoem"]:
        for prefix in ["pdb", "chain", "cov", "res"]:
            reps[f"{prefix}_{col}"] = ""
    names = ds.write_manifest(reps, facts, path=str(tmp_path / "p.csv"))[
        "protein_name"
    ].tolist()
    assert names == [
        "Large ribosomal subunit protein uL2 (P1)",
        "Large ribosomal subunit protein uL2 (P2)",
        "Ubiquitin",
    ]


# ── what the chosen structures are (read from the files) ────────────────────


def test_describe_structures_and_composition(tmp_path):
    from tests.synthetic import build_backbone, protein_like_torsions, write_cif

    phi, psi = protein_like_torsions(40, seed=1)
    exp = tmp_path / "data" / "experimental"
    exp.mkdir(parents=True)
    write_cif(exp / "1NMR.cif", [build_backbone(phi, psi), build_backbone(phi + 2, psi)])
    write_cif(exp / "2EMX.cif", [build_backbone(phi, psi)])
    manifest = pd.DataFrame(
        [
            {"protein_name": "A", "uniprot_id": "PA", "pdb_xray": "", "chain_xray": "", "pdb_nmr": "1NMR",
             "chain_nmr": "A", "pdb_cryoem": "2EMX", "chain_cryoem": "A"},
            {"protein_name": "B", "uniprot_id": "PB", "pdb_xray": "3XRY", "chain_xray": "A", "pdb_nmr": "1NMR",
             "chain_nmr": "A", "pdb_cryoem": "2EMX", "chain_cryoem": "A"},
        ]
    )
    described = ds.describe_structures(manifest, str(tmp_path / "data"))
    assert len(described) == 5
    assert (described["status"] == "missing").sum() == 1  # 3XRY is not on disk
    nmr = described[(described["method"] == "NMR") & (described["status"] == "ok")].iloc[0]
    assert nmr["n_models"] == 2 and nmr["n_residues"] == 40
    assert not described["before_af2_cutoff"].any()  # synthetic files carry no dates
    combos = ds.composition(manifest).set_index("experimental methods")["proteins"].to_dict()
    assert combos == {"NMR + Cryo-EM": 1, "X-ray + NMR + Cryo-EM": 1}
