import make_figures

make_figures.extract_image_psnrs(dict({'base': '/home/david/Research/Tensor/siren/logs/experiment_7/summaries/'}))

make_figures.extract_images_from_summary('/home/david/Research/Tensor/siren/logs/experiment_7/summaries/', 'train_pred_image', suffix='', img_outdir='out/', colormap=None)