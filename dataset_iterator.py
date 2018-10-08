def make_dataset():
    """Each time make_dataset is called, it returns a new frame, camera batch of size 12, randomly drawn from data directory"""
    
    # First 130 scenes. 12 facing away from wall.
    list1 = np.arange(12).tolist()
    views1 = [[-1., -1., 0., 0.707, 0.707, 0., 1.],
            [-0.5, -1., 0., 1., 0., 0., 1.],
            [0.5, -1., 0., 1., 0., 0., 1.],
            [1., -1., 0., 0.707, -0.707, 0., 1.],
            [1., -0.5, 0., 0., -1., 0., 1.],
            [1., 0.5, 0., 0., -1., 0., 1.],
            [1., 1., 0., -0.707, -0.707, 0., 1.],
            [0.5, 1., 0., -1, 0., 0., 1.],
            [-0.5, 1., 0., -1., 0., 0., 1.],
            [-1., 1., 0., -0.707, 0.707, 0., 1.],
            [-1., 0.5, 0., 0., 1., 0., 1.],
            [-1, -0.5, 0., 0., 1., 0., 1.]]

    # Scenes 131 and up. 8 center facing.
    list1 = np.arange(8).tolist()
    views2 = [[-1., -1., 0., 0.707, 0.707, 0., 1.],
            [0., -1., 0., 1., 0., 0., 1.],
            [1., -1., 0., 0.707, -0.707, 0., 1.],
            [1., 0., 0., 0., -1., 0., 1.],
            [1., 1., 0., -0.707, -0.707, 0., 1.],
            [0., 1., 0., -1., 0., 0., 1.],
            [-1., 1., 0., -0.707, 0.707, 0., 1.],
            [-1., 0., 0., 0., 1., 0., 1.]]

    for batch in range(300):
        final_frames = []
        final_viewpoints = []
        frames = np.ndarray(shape=(12, 6, 64, 64, 3), dtype=float)
        viewpoints = np.ndarray(shape=(12, 6, 7), dtype=float)
        folder_choices = np.random.choice(235, 12, replace=False)

        for element in folder_choices:

            if element < 129:
                file_choices = []
                view_choices = []
                name = "./fruit_stills/Scene" + str(element+1) + "/*"
                list_of_files = glob(name)
                index = np.random.choice(12, 6, replace=False).tolist()
                for nn in index:
                    nn = int(nn)
                    view_choices.append(views1[nn])
                    file_choices.append(list_of_files[nn])
                frames = np.asarray([imread(file) for file in file_choices])
                final_frames.append(frames)
                viewpoints = np.asarray(view_choices)
                final_viewpoints.append(viewpoints)

            else:
                file_choices = []
                view_choices = []
                name = "./fruit_stills_dataset/Scene" + str(element+1) + "/*"
                list_of_files = glob(name)
                index = np.random.choice(8, 6, replace=False).tolist()
                for nnn in index:
                    nnn = int(nnn)
                    view_choices.append(views2[nnn])
                    file_choices.append(list_of_files[nnn])
                frames = np.asarray([imread(file) for file in file_choices])
                final_frames.append(frames)
                viewpoints = np.asarray(view_choices)
                final_viewpoints.append(viewpoints)


        final_frames = np.asarray(final_frames)
        final_frames = final_frames.astype(np.float32)
        frames = 2 * ((1/255) * final_frames) - 1
        frames = frames.astype(np.float32)
        viewpoints = np.asarray(final_viewpoints)
        viewpoints = viewpoints.astype(np.float32)
        frames = tf.convert_to_tensor(frames, np.float32)
        viewpoints = tf.convert_to_tensor(viewpoints, np.float32)
        yield frames, viewpoints
