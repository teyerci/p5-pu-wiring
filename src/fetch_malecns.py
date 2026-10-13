"""MaleCNS (neuPrint male-cns:v1.0) fetch functions. NOT YET RUN.

Blocked on two things:
  1. a neuPrint token in NEUPRINT_APPLICATION_CREDENTIALS (env or gitignored .env);
  2. the bulk-download addresses for the edge tables, which Project 4 puts in
     its README for everyone (Data Conventions, Data sources).

Mapping per the Data Conventions "Source column mapping"; names marked
"(confirm)" there must be checked against the printed columns before use.
"""
from __future__ import annotations

import config

DATASET = "malecns"
RELEASE = config.MALECNS_DATASET.split(":")[1]          # v1.0


def fetch_neurons():
    raise NotImplementedError("MaleCNS neurons: needs a neuPrint token; see module docstring")


def fetch_edges():
    raise NotImplementedError("MaleCNS edges: needs the bulk-download address from Project 4's README")


def fetch_edgeroi():
    raise NotImplementedError("MaleCNS edgeroi: needs the bulk-download address from Project 4's README")
