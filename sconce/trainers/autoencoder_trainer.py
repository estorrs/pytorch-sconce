from .basic_trainer import BasicTrainer
from matplotlib import pyplot as plt


class AutoencoderTrainer(BasicTrainer):
    def plot_input_output_pairs(self, title='A Sampling of Autoencoder Results',
        num_cols=10, figsize=(15, 3.2)):
        data, labels = self.test_data_generator.next()
        out_dict = self._run_model(data, labels, train=True)
        x_out = out_dict['x_out']

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)

        for i in range(num_cols):
            input_image = data[i][0]
            output_image = x_out.view_as(data).data.cpu()[i][0]

            ax = fig.add_subplot(2, num_cols, i+1)
            ax.imshow(input_image, cmap='gray')
            if i == 0:
                ax.set_ylabel('Input')
            else:
                ax.axis('off')

            ax = fig.add_subplot(2, num_cols, num_cols+i+1)
            ax.imshow(output_image, cmap='gray')
            if i == 0:
                ax.set_ylabel('Output')
            else:
                ax.axis('off')
        return fig

    def plot_latent_space(self, title="Latent Representation", figsize=(8, 8)):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)

        self.model.train(False)
        for data, labels in self.test_data_loader:
            x_in = Variable(data).cuda()
            y_in = Variable(labels).cuda()
            in_dict = {'x_in': x_in, 'y_in': y_in}

            x_latent = self.model.encode(**in_dict)

            x_latent_numpy = x_latent.cpu().data.numpy()
            plt.scatter(x=x_latent_numpy.T[0], y=x_latent_numpy.T[1],
                        c=labels.numpy(), alpha=0.4)
        plt.colorbar()
        return fig
