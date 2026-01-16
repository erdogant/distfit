import numpy as np
import unittest
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib
import pytest
from scipy.stats import kstest
from distfit import distfit
matplotlib.use('Agg')  # Use non-interactive backend for tests


def test_fit_creates_marginals_and_corr():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)
    # Checks
    assert hasattr(dfit.model, "corr")
    assert dfit.model.corr.shape[0] == dfit.model.corr.shape[1]
    assert len(dfit.model.marginals) == dfit.model.corr.shape[0]


def test_correlation_matrix_properties():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)

    corr = dfit.model.corr
    # Symmetry
    assert np.allclose(corr, corr.T)

    # Diagonal must be ones
    assert np.allclose(np.diag(corr), 1.0)

    # Positive semi-definite
    eigvals = np.linalg.eigvalsh(corr)
    assert np.all(eigvals > -1e-8)



def test_evaluate_pdf_output():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)

    out = dfit.evaluate_pdf(X)

    assert isinstance(out, dict)
    assert "copula_density" in out
    assert "score" in out

    p = out["copula_density"]
    assert p.ndim == 1
    assert len(p) == len(X)
    assert np.all(p > 0)
    assert np.isfinite(out["score"])


def test_score_equals_mean_log_density():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)
    
    out = dfit.evaluate_pdf(X)
    logp = np.log(out["copula_density"])
    assert np.isclose(out["score"], np.mean(logp))


def test_probability_integral_transform_uniformity():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)

    X = multivariate_data()
    d = X.shape[1]

    for i, m in enumerate(dfit.model.marginals):
        dist = dfit.model.marginals[m].model["model"]
        U = dist.cdf(X[:, i])

        # KS test against Uniform(0,1)
        stat, pval = kstest(U, "uniform")
        assert pval > 0.01


def test_generate_shape_and_finiteness():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)

    Xgen = dfit.generate(n=500)

    assert Xgen.shape[0] == 500
    assert Xgen.shape[1] == len(dfit.model.marginals)
    assert np.all(np.isfinite(Xgen))


def test_generate_preserves_dependence():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)

    Xgen = dfit.generate(n=3000)

    corr_gen = np.corrcoef(Xgen, rowvar=False)
    corr_fit = dfit.model.corr

    # Allow noise but expect structure
    assert np.allclose(
        np.tril(corr_gen, -1),
        np.tril(corr_fit, -1),
        atol=0.15,
    )



def test_predict_outliers_returns_boolean_mask():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)

    outliers = dfit.predict_outliers(X)

    assert outliers.dtype == bool
    assert len(outliers) == len(X)
    assert outliers.sum() > 0



def test_outliers_have_lower_density():
    dfit = distfit(multivariate=True, verbose=False)
    X = multivariate_data()
    dfit.fit_transform(X)

    pdf = dfit.evaluate_pdf(X)["copula_density"]
    outliers = dfit.predict_outliers(X)

    assert np.mean(pdf[outliers]) < np.mean(pdf[~outliers])


def test_high_dimensional_fit():
    dfit = distfit(multivariate=True, verbose=False)    
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    dfit.fit_transform(X)

    out = dfit.evaluate_pdf(X)
    assert len(out["copula_density"]) == len(X)



def multivariate_data():
    rng = np.random.default_rng(42)
    mean = [0, 0]
    cov = [[1, 0.6],
           [0.6, 1]]
    return rng.multivariate_normal(mean, cov, size=2000)

