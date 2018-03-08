from .basic_trainer import BasicTrainer


class ClassifierTrainer(BasicTrainer):
    def get_classification_accuracy(self):
        num_correct = 0
        for data, labels in self.test_data_loader:
            out_dict = self._run_model(data, labels, train=False)
            y_out = np.argmax(out_dict['y_out'].cpu().data.numpy(), axis=1)
            y_in = out_dict['y_in'].cpu().data.numpy()
            num_correct += (y_out - y_in == 0).sum()
        return num_correct / len(self.test_data_loader.dataset)

    def plot_confusion_matrix(self):
        pass

    def plot_misclassified_samples(self, true_label):
        pass
