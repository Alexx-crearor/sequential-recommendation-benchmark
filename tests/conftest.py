import pytest
import pandas as pd
from src.data.preprocessing import preprocess
from src.data.splitting import leave_one_out


@pytest.fixture
def frame():
    # Dense 6x6 fixture with chronology rotated to produce distinct held-outs.
    raw = pd.DataFrame([{"user": f"u{u}", "item": f"i{(u+t)%6}", "timestamp": t, "rating": 5}
                        for u in range(6) for t in range(6)])
    return preprocess(raw)[0]


@pytest.fixture
def parts(frame):
    return leave_one_out(frame)
