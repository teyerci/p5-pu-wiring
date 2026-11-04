"""Project code. String columns are held in Arrow memory: the edgeroi tables have tens of
millions of rows, and as Python objects they stall a machine short of memory."""
import pandas as pd

pd.set_option("mode.string_storage", "pyarrow")
