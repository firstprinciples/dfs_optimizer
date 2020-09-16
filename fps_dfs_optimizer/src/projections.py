from scipy.stats import norm


def get_projections(df):
    ppds = norm(5, 1).rvs(size=len(df))
    vpds = norm(2, 0.5).rvs(size=len(df))
    points = ppds * df['salary'].values / 1000
    std = vpds * df['salary'].values / 1000
    return points, std
    