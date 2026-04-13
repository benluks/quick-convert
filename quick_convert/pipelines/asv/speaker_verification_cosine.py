from __future__ import annotations

import os

import speechbrain as sb
import torch
import torchaudio
from speechbrain.dataio import audio_io
from tqdm import tqdm


def compute_embedding(wavs, wav_lens, params):
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader, params, run_opts):
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(run_opts["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = any(seg_id not in embedding_dict for seg_id in seg_ids)
            if not found:
                continue

            wavs = wavs.to(run_opts["device"])
            lens = lens.to(run_opts["device"])
            emb = compute_embedding(wavs, lens, params).unsqueeze(1)

            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()

    return embedding_dict


def get_verification_scores(
    veri_test,
    params,
    enrol_dict,
    test_dict,
    train_dict=None,
):
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    train_cohort = None
    if "score_norm" in params:
        if train_dict is None:
            raise ValueError("score_norm requested but train_dict is None")
        train_cohort = torch.stack(list(train_dict.values()))

    with open(save_file, "w", encoding="utf-8") as s_file:
        for line in veri_test:
            parts = line.split()
            lab_pair = int(parts[0].rstrip().split(".")[0].strip())
            enrol_id = parts[1].rstrip().split(".")[0].strip()
            test_id = parts[2].rstrip().split(".")[0].strip()

            enrol = enrol_dict[enrol_id]
            test = test_dict[test_id]

            if train_cohort is not None:
                enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
                score_e_c = similarity(enrol_rep, train_cohort)

                if "cohort_size" in params:
                    score_e_c = torch.topk(score_e_c, k=params["cohort_size"], dim=0)[0]

                mean_e_c = torch.mean(score_e_c, dim=0)
                std_e_c = torch.std(score_e_c, dim=0)

                test_rep = test.repeat(train_cohort.shape[0], 1, 1)
                score_t_c = similarity(test_rep, train_cohort)

                if "cohort_size" in params:
                    score_t_c = torch.topk(score_t_c, k=params["cohort_size"], dim=0)[0]

                mean_t_c = torch.mean(score_t_c, dim=0)
                std_t_c = torch.std(score_t_c, dim=0)

            score = similarity(enrol, test)[0]

            if train_cohort is not None:
                if params["score_norm"] == "z-norm":
                    score = (score - mean_e_c) / std_e_c
                elif params["score_norm"] == "t-norm":
                    score = (score - mean_t_c) / std_t_c
                elif params["score_norm"] == "s-norm":
                    score_e = (score - mean_e_c) / std_e_c
                    score_t = (score - mean_t_c) / std_t_c
                    score = 0.5 * (score_e + score_t)

            s_file.write(f"{enrol_id} {test_id} {lab_pair} {float(score)}\n")

            if lab_pair == 1:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

    return positive_scores, negative_scores


def dataio_prep(params):
    data_folder = params["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"],
        replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        select_n=params.get("n_train_snts"),
    )

    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"],
        replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = audio_io.load(wav, num_frames=num_frames, frame_offset=start)

        if fs != params["sample_rate"]:
            sig = torchaudio.functional.resample(sig, fs, params["sample_rate"])

        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, enrol_dataloader, test_dataloader
