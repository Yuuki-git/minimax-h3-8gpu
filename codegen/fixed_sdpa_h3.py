from ninetoothed import language as ntl


def application_without_kv_cache(query, key, value, scale, output):
    for i in range(query.shape[0]):
        query_i = (1.4426950408889634 * ntl.cast(scale, ntl.float32) * query[i]).to(query[i].dtype)
        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        max = ntl.full((query_i.shape[-2],), float('-inf'), dtype=ntl.float32)
        for j in range(key.shape[0]):
            qk = ntl.dot(query_i, ntl.trans(key[j]))
            qk = ntl.where(key[j].offsets(-2) < key.source.shape[-2], qk, float('-inf'))
            next_max = ntl.maximum(max, ntl.max(qk, 1))
            stable_qk = ntl.exp2(qk - next_max[:, None])
            alpha = ntl.exp2(max - next_max)
            acc = acc * alpha[:, None] + ntl.dot(stable_qk.to(value[i].dtype), value[j])
            max = next_max
            lse = lse * alpha + ntl.sum(stable_qk, 1)
        acc /= lse[:, None]
        output[i] = acc
