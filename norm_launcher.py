from time import time
import os
import torch
from scTC import scTC
from single_cell_tools import *
import numpy as np
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# for repeatability
setup_seed(26)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=10, type=int,
                        help='number of clusters')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='./data/worm_neuron_cell.h5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--sigma', default=2.5, type=float,
                        help='coefficient of random noise')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float,
                        help='tolerance for delta clustering labels to terminate training stage')
    parser.add_argument('--ae_weights', default=None,
                        help='file to pretrained weights, None for a new pretraining')
    parser.add_argument('--save_dir', default='results/scTC_worm/',
                        help='directory to save model weights during the training stage')
    parser.add_argument('--ae_weight_file', default='AE_weights_worm.pth.tar',
                        help='file name to save model weights after the pretraining stage')
    parser.add_argument('--final_latent_file', default='final_latent_file_worm.txt',
                        help='file name to save final latent representations')
    parser.add_argument('--predict_label_file', default='pred_labels_worm.txt',
                        help='file name to save final clustering labels')
    parser.add_argument('--volume', default=400, type=int)
    parser.add_argument('--generate',default=30000, type=int)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()

    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(args)
    print(adata.X.shape)
    print(y.shape)


    anchor, positive, negative = generate_triplets(y, generate=args.generate, volume=args.volume)
    print('generate triplets : %d' % args.generate)

    model = scTC(input_dim=adata.n_vars, z_dim=32,
                 encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma,
                 device=args.device)

    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   batch_size=args.batch_size, epochs=args.pretrain_epochs,
                                   ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   n_clusters=args.n_clusters, init_centroid=None, anchor=anchor, positive=positive, negative=negative,
                                   y_pred_init=None, y=y, batch_size=args.batch_size, num_epochs=args.maxiter,
                                   update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)


    print('Total time: %d seconds.' % int(time() - t0))

    if y is not None:
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print('Evaluating cells: NMI= %.4f, ARI= %.4f,ACC= %.4f' % (nmi, ari, acc))

    final_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float32)).cpu().numpy()
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
    np.savetxt(args.predict_label_file, y_pred, delimiter=",", fmt="%i")