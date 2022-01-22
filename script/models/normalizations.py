import numpy as np

def id_post_proc(trace):
    return trace

def instance_local_scaling(trace):
    mini = np.min(trace)
    maxi = np.max(trace)
    half_span = (maxi - mini) / 2
    return (trace - mini - half_span) / half_span

def make_instance_global_scaling(mini,maxi):
    half_span = (maxi - mini) / 2
    def scaling(trace):
        return (trace - mini - half_span) / half_span
    return scaling

def make_instance_global_stand(mean,stddev):
    def scaling(trace):
        return (trace - mean) / stddev
    return scaling

def instance_local_stand(trace):
    mean = np.mean(trace)
    stddev = np.sqrt(np.var(trace))
    return (trace - mean) / stddev


def make_feature_scaling(minis,maxis):
    half_spans = (maxis - minis) / 2
    def scaling(trace):
        return (trace - minis - half_spans) / half_spans
    return scaling

def make_feature_stand(means,stddevs):
    def scaling(trace):
        return (trace - means) / stddevs
    return scaling

