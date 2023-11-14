
if __name__ == '__main__':

    opt = TrainOptions().parse()   #获取一些训练参数

    dataset = create_dataset(opt)  #创建数据集

    dataset_size = len(dataset)    #数据集大小

    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      #创建模型

    model.setup(opt)               #模型初始化

    visualizer = Visualizer(opt)   #可视化函数

    total_iters = 0                #迭代batch次数

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        epoch_iter = 0                  #当前epoch迭代batch数

        for i, data in enumerate(dataset):  #每一个epoch内层循环

            visualizer.reset()

            total_iters += opt.batch_size #总迭代batch数

            epoch_iter += opt.batch_size

            model.set_input(data)         #输入数据

            model.optimize_parameters()   #迭代更新

            if total_iters % opt.display_freq == 0:   #visdom可视化

                save_result = total_iters % opt.update_html_freq == 0

                model.compute_visuals()

                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    #存储损失等信息

                losses = model.get_current_losses()

                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                if opt.display_id > 0:

                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   #存储模型

                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))

                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'

                model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0: #每隔opt.save_epoch_freq各epoch存储模型

            model.save_networks('latest')

            model.save_networks(epoch)

        model.update_learning_rate()#每一个epoch后更新学习率

其中的一些重要训练参数配置如下：

input_nc=1，表示生成器输入为1通道图像，即L通道。

output_nc=2，表示生成器输出为2通道图像，即AB通道。

ngf=64，表示生成器最后1个卷积层输出通道为64。

ndf=64，表示判别器最后1个卷积层输出通道为64。

n_layers_D=3，表示使用默认的PatchGAN，它相当于对70×70大小的图像块进行判别。

norm=batch，batch_size=1，表示使用批次标准化。

load_size=286，表示载入的图像尺寸。

crop_size=256，表示图像裁剪即训练尺寸。
