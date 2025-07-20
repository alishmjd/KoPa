import torch


def load_pretrain_kge(path):
    kge_model = torch.load(path)
    ent_embs = torch.tensor(kge_model["ent_embeddings.weight"]).cpu()
    rel_embs = torch.tensor(kge_model["rel_embeddings.weight"]).cpu()
    ent_embs.requires_grad = False
    rel_embs.requires_grad = False
    ent_dim = ent_embs.shape[1]
    rel_dim = rel_embs.shape[1]
    print(ent_dim, rel_dim)
    if ent_dim != rel_dim:
        rel_embs = torch.cat((rel_embs, rel_embs), dim=-1)
    return ent_embs, rel_embs
