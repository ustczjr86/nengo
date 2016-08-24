from copy import copy
import warnings

import numpy as np
import pytest

import nengo
from nengo.exceptions import NetworkContextError, NotAddedToNetworkWarning
from nengo.params import iter_params
from nengo.utils.compat import is_array_like, pickle
from nengo.utils.testing import warns


@pytest.fixture(scope='module', autouse=True)
def error_on_not_added_to_network_warning(request):
    catcher = warnings.catch_warnings(record=True)
    catcher.__enter__()
    warnings.simplefilter(
        'error', category=NotAddedToNetworkWarning)
    request.addfinalizer(lambda: catcher.__exit__(None, None, None))


def assert_is_copy(cp, original):
    assert cp is not original  # ensures separate parameters
    for param in iter_params(cp):
        param_inst = getattr(cp, param)
        if isinstance(param_inst, nengo.solvers.Solver) or isinstance(
                param_inst, nengo.base.NengoObject):
            assert param_inst is getattr(original, param)
        elif is_array_like(param_inst):
            assert np.all(param_inst == getattr(original, param))
        else:
            assert param_inst == getattr(original, param)


def assert_is_deepcopy(cp, original):
    assert cp is not original  # ensures separate parameters
    for param in iter_params(cp):
        param_inst = getattr(cp, param)
        if isinstance(param_inst, nengo.solvers.Solver) or isinstance(
                param_inst, nengo.base.NengoObject):
            assert_is_copy(param_inst, getattr(original, param))
        elif is_array_like(param_inst):
            assert np.all(param_inst == getattr(original, param))
        else:
            assert param_inst == getattr(original, param)


def make_ensemble():
    with nengo.Network():
        e = nengo.Ensemble(10, 1, radius=2.)
    return e


def test_neurons_reference_copy():
    original = make_ensemble()
    cp = original.copy(add_to_container=False)
    assert original.neurons.ensemble is original
    assert cp.neurons.ensemble is cp


def make_probe():
    with nengo.Network():
        e = nengo.Ensemble(10, 1)
        p = nengo.Probe(e, synapse=0.01)
    return p


def make_node():
    with nengo.Network():
        n = nengo.Node(np.min, size_in=2, size_out=2)
    return n


def make_connection():
    with nengo.Network():
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(10, 1)
        c = nengo.Connection(e1, e2, transform=2.)
    return c


def make_network():
    with nengo.Network() as model:
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(10, 1)
        nengo.Connection(e1, e2, transform=2.)
        nengo.Probe(e2)
    return model


def test_copy_in_network_default_add():
    original = make_network()

    with nengo.Network() as model:
        cp = original.copy()
    assert cp in model.all_objects

    assert_is_deepcopy(cp, original)


def test_copy_outside_network_default_add():
    original = make_network()
    cp = original.copy()
    assert_is_deepcopy(cp, original)


def test_network_copies_defaults():
    original = nengo.Network()
    original.config[nengo.Ensemble].radius = 1.5
    original.config[nengo.Connection].synapse = nengo.Lowpass(0.1)

    cp = original.copy()
    assert cp.config[nengo.Ensemble].radius == 1.5
    assert cp.config[nengo.Connection].synapse == nengo.Lowpass(0.1)


def test_network_copy_allows_independent_manipulation():
    with nengo.Network() as original:
        nengo.Ensemble(10, 1)
    original.config[nengo.Ensemble].radius = 1.

    cp = original.copy()
    with cp:
        e2 = nengo.Ensemble(10, 1)
    cp.config[nengo.Ensemble].radius = 2.

    assert e2 not in original.ensembles
    assert original.config[nengo.Ensemble].radius == 1.


def test_copies_structure():
    with nengo.Network() as original:
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(10, 1)
        nengo.Connection(e1, e2)
        nengo.Probe(e2)

    cp = original.copy()

    assert cp.connections[0].pre is not e1
    assert cp.connections[0].post is not e2
    assert cp.connections[0].pre in cp.ensembles
    assert cp.connections[0].post in cp.ensembles

    assert cp.probes[0].target is not e2
    assert cp.probes[0].target in cp.ensembles


def test_network_copy_builds(RefSimulator):
    RefSimulator(make_network().copy())


@pytest.mark.parametrize(('make_f', 'assert_f'), [
    (make_ensemble, assert_is_copy),
    (make_probe, assert_is_copy),
    (make_node, assert_is_copy),
    (make_connection, assert_is_copy),
    (make_network, assert_is_copy),
])
class TestCopy(object):
    """A basic set of tests that should pass for all objects."""

    def test_copy_in_network(self, make_f, assert_f):
        original = make_f()

        with nengo.Network() as model:
            cp = original.copy(add_to_container=True)
        assert cp in model.all_objects

        assert_f(cp, original)

    def test_copy_in_network_without_adding(self, make_f, assert_f):
        original = make_f()

        with nengo.Network() as model:
            cp = original.copy(add_to_container=False)
        assert cp not in model.all_objects

        assert_f(cp, original)

    def test_copy_outside_network(self, make_f, assert_f):
        original = make_f()
        with pytest.raises(NetworkContextError):
            original.copy(add_to_container=True)

    def test_copy_outside_network_without_adding(self, make_f, assert_f):
        original = make_f()
        cp = original.copy(add_to_container=False)
        assert_f(cp, original)

    def test_python_copy_warns_abt_adding_to_network(self, make_f, assert_f):
        original = make_f()
        copy(original)  # Fine because not in a network
        with nengo.Network():
            with warns(NotAddedToNetworkWarning):
                copy(original)


@pytest.mark.parametrize('make_f', (
    make_ensemble, make_probe, make_node, make_connection, make_network
))
class TestPickle(object):
    """A basic set of tests that should pass for all objects."""

    def test_pickle_roundtrip(self, make_f):
        original = make_f()
        cp = pickle.loads(pickle.dumps(original))
        assert_is_deepcopy(cp, original)

    def test_unpickling_warning_in_network(self, make_f):
        original = make_f()
        pkl = pickle.dumps(original)
        with nengo.Network():
            with warns(NotAddedToNetworkWarning):
                pickle.loads(pkl)


@pytest.mark.parametrize('original', [
    nengo.learning_rules.PES(),
    nengo.learning_rules.BCM(),
    nengo.learning_rules.Oja(),
    nengo.learning_rules.Voja(),
    nengo.processes.WhiteNoise(),
    nengo.processes.FilteredNoise(),
    nengo.processes.BrownNoise(),
    nengo.processes.PresentInput([.1, .2], 1.),
    nengo.synapses.LinearFilter([.1, .2], [.3, .4], True),
    nengo.synapses.Lowpass(0.005),
    nengo.synapses.Alpha(0.005),
    nengo.synapses.Triangle(0.005),
])
class TestFrozenObjectCopies(object):

    def test_copy(self, original):
        assert_is_copy(copy(original), original)

    def test_pickle_roundtrip(self, original):
        assert_is_deepcopy(pickle.loads(pickle.dumps(original)), original)