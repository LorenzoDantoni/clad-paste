import os

import numpy as np
import torch
from tqdm import tqdm

from src.models.stfpm_add.loss import STFPMLoss
from src.utilities.utility_ad import standardize_scores, test_epoch_anomaly_maps
from src.utilities.utility_plot import plot_predict


class Trainer_STFPM_paste:

    def __init__(self, strategy, ad_model):
        self.strategy = strategy
        self.device = strategy.device
        self.ad_model = ad_model
        self.batch_size = strategy.parameters['batch_size']
        self.lr = self.strategy.lr

        self.optimizer = torch.optim.SGD(
            params=self.ad_model.student.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.strategy.parameters.get('weight_decay', 1e-4),
        )

        self.loss_fcn = STFPMLoss()

        self.sample_strategy = self.strategy.parameters.get("sample_strategy")


    def train_epoch(self, dataloader):
        self.ad_model.training = True
        l_fastflow_loss = 0.0
        dataSize = len(dataloader.dataset)
        lista_indices = []

        self.ad_model.sliced_teacher.eval()
        self.ad_model.teacher.eval()
        self.ad_model.student.train()

        batch_index = 0
        for batch in tqdm(dataloader):
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0], self.sample_strategy)

            x = batch[0]
            batch_size = x.size(0)
            indices = batch[2]
            lista_indices.extend(indices.detach().cpu().numpy())

            self.optimizer.zero_grad()
            x = x.to(self.ad_model.device)

            teacher_features, student_features = self.ad_model.forward(x, self.sample_strategy, self.strategy.index_training, test=False)
            loss = self.loss_fcn(teacher_features, student_features)
            loss.backward()

            self.optimizer.step()
            l_fastflow_loss += loss.item() * batch_size
            # self.scheduler.step()
            batch_index += 1

        '''
        torch.save(self.ad_model.teacher, os.path.join(self.strategy.train_output_dir,
                                            'teacher_tmp.pth'))
        torch.save(self.ad_model.student, os.path.join(self.strategy.train_output_dir,
                                            'student_tmp.pth'))
        '''

        if self.strategy.parameters['early_stopping'] == True:
            run_name1 = 'model_student' + str(self.strategy.current_epoch)
            torch.save(self.ad_model.student.state_dict(), os.path.join(self.strategy.checkpoints, run_name1 + ".pckl"))

        l_fastflow_loss /= dataSize
        lista_indices = np.asarray(lista_indices)

        metrics_epoch = {"loss": l_fastflow_loss}
        other_data_epoch = {"indices": lista_indices}

        return metrics_epoch, other_data_epoch

    def test_epoch(self, dataloader):
        dataset = self.strategy.complete_test_dataset
        self.ad_model.training = False
        lista_indices = []
        losses, l_anomaly_maps, lista_labels = [], [], []
        test_imgs, gt_list, gt_mask_list = [], [], []
        batch_index = 0

        self.ad_model.teacher.eval()
        self.ad_model.student.eval()

        for batch in tqdm(dataloader):
            masks = []
            data, indices, anomaly_info = batch[0], batch[2], batch[3]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy())
            data = data.to(self.ad_model.device)

            with torch.no_grad():
                anomaly_maps = self.ad_model.forward(data, self.sample_strategy, self.strategy.index_training, test=True)

            heatmap = anomaly_maps[:, 0].detach().cpu().numpy()
            # print(f"Heatmap size: {heatmap.shape}")
            # heatmap = torch.mean(heatmap, dim=1)
            l_anomaly_maps.extend(heatmap)

            # lista_labels.extend(class_ids)
            lista_labels.extend(class_ids.detach().cpu().numpy())

            for i, idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path, anomaly_info[i])
                masks.append(mask)

            mask = torch.stack(masks)
            # test_imgs.extend(data.detach().cpu().numpy())
            gt_list.extend(anomaly_info.detach().cpu().numpy())
            gt_mask_list.extend(mask.detach().cpu().numpy())

            batch_index += 1

        # heatmaps = upsample(heatmaps, size=data.size(2), mode='bilinear')
        # heatmaps = gaussian_smooth(heatmaps, sigma=4)

        ###gt_mask = np.asarray(gt_mask_list)
        # scores = rescale(heatmaps)

        l_anomaly_maps = standardize_scores(l_anomaly_maps)
        # threshold = get_threshold(gt_mask, scores)

        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps, gt_mask_list, gt_list, self.strategy.index_training,
                                               self.strategy.run,
                                               self.strategy.labels_map[self.strategy.index_training],
                                               self.strategy.index_training, self.strategy.path_logs)
        diz_metriche["loss"] = 1 - diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]

        mode = self.strategy.trainer.mode if hasattr(self.strategy.trainer, 'mode') else "reconstruct"

        metrics_epoch = diz_metriche
        other_data_epoch = {}

        return metrics_epoch, other_data_epoch


    def evaluate_data(self, dataloader, test_loss_function=None):
        dataset = self.strategy.complete_test_dataset
        test_task_index = self.strategy.current_test_task_index
        index_training = self.strategy.index_training
        self.ad_model.training = False
        lista_indices = []
        losses, l_anomaly_maps, lista_labels = [], [], []
        test_imgs, gt_list, gt_mask_list = [], [], []

        self.ad_model.teacher.eval()
        self.ad_model.student.eval()

        for batch in tqdm(dataloader):
            masks = []
            data, indices, anomaly_info = batch[0], batch[2], batch[3]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy())
            data = data.to(self.ad_model.device)

            with torch.no_grad():
                anomaly_maps = self.ad_model.forward(data, self.sample_strategy, self.strategy.index_training, test=True)

            heatmap = anomaly_maps[:, 0].detach().cpu().numpy()
            # print(f"Heatmap size: {heatmap.shape}")
            # heatmap = torch.mean(heatmap, dim=1)
            l_anomaly_maps.extend(heatmap)

            lista_labels.extend(class_ids.detach().cpu().numpy())

            for i, idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path, anomaly_info[i])
                masks.append(mask)

            mask = torch.stack(masks)
            test_imgs.extend(data.detach().cpu().numpy())
            gt_list.extend(anomaly_info.detach().cpu().numpy())
            gt_mask_list.extend(mask.detach().cpu().numpy())

        # heatmaps = upsample(heatmaps, size=data.size(2), mode='bilinear')
        # heatmaps = gaussian_smooth(heatmaps, sigma=4)

        ###gt_mask = np.asarray(gt_mask_list)
        # scores = rescale(heatmaps)

        l_anomaly_maps = standardize_scores(l_anomaly_maps)
        # threshold = get_threshold(gt_mask, scores)

        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps, gt_mask_list, gt_list, self.strategy.index_training,
                                               self.strategy.run,
                                               self.strategy.labels_map[self.strategy.index_training],
                                               self.strategy.index_training, self.strategy.path_logs)
        diz_metriche["loss"] = 1 - diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]

        # added
        if self.strategy.index_training == 9:
            plot_predict(self, lista_labels, l_anomaly_maps, gt_mask_list, lista_indices, threshold, test_imgs)

        metrics_epoch = diz_metriche
        other_data_epoch = {}
        return metrics_epoch, other_data_epoch
