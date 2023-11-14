
class ColorizationDataset(BaseDataset):

   def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.dir = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))

        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')

        self.transform = get_transform(self.opt, convert=False)

    def __getitem__(self, index):

        path = self.AB_paths[index]

        im = Image.open(path).convert('RGB') ## 读取RGB图

        im = self.transform(im) ## 进行预处理

        im = np.array(im)

        lab = color.rgb2lab(im).astype(np.float32) ## 将RGB图转换为CIELab图

        lab_t = transforms.ToTensor()(lab)

        L = lab_t[[0], ...] / 50.0 - 1.0 ## 将L通道(index=0)数值归一化到-1到1之间

        AB = lab_t[[1, 2], ...] / 110.0 ## 将A，B通道(index=1,2)数值归一化到0到1之间

        return {'A': L, 'B': AB, 'A_paths': path, 'B_paths': path}

