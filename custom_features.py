# normalized features_ground
asl.df['grnd-lx-mean']= asl.df['speaker'].map(df_means['grnd-lx'])
asl.df['grnd-ly-mean']= asl.df['speaker'].map(df_means['grnd-ly'])
asl.df['grnd-rx-mean']= asl.df['speaker'].map(df_means['grnd-rx'])
asl.df['grnd-ry-mean']= asl.df['speaker'].map(df_means['grnd-ry'])

asl.df['grnd-lx-std']= asl.df['speaker'].map(df_std['grnd-lx'])
asl.df['grnd-ly-std']= asl.df['speaker'].map(df_std['grnd-ly'])
asl.df['grnd-rx-std']= asl.df['speaker'].map(df_std['grnd-rx'])
asl.df['grnd-ry-std']= asl.df['speaker'].map(df_std['grnd-ry'])

asl.df['norm-grx'] = (asl.df['grnd-rx'] - asl.df['grnd-rx-mean']) / asl.df['grnd-rx-std']
asl.df['norm-gry'] = (asl.df['grnd-ry'] - asl.df['grnd-ry-mean']) / asl.df['grnd-ry-std']
asl.df['norm-glx'] = (asl.df['grnd-lx'] - asl.df['grnd-lx-mean']) / asl.df['grnd-lx-std']
asl.df['norm-gly'] = (asl.df['grnd-ly'] - asl.df['grnd-ly-mean']) / asl.df['grnd-ly-std']

# TODO define a list named 'features_custom' for building the training set
features_custom = ['norm-grx', 'norm-gry', 'norm-glx','norm-gly']

# normalized features_polar
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

asl.df['polar-rr-mean'] = asl.df['speaker'].map(df_means['polar-rr'])
asl.df['polar-rt-mean'] = asl.df['speaker'].map(df_means['polar-rtheta'])
asl.df['polar-lr-mean'] = asl.df['speaker'].map(df_means['polar-lr'])
asl.df['polar-lt-mean'] = asl.df['speaker'].map(df_means['polar-ltheta'])

asl.df['polar-rr-std'] = asl.df['speaker'].map(df_std['polar-rr'])
asl.df['polar-rt-std'] = asl.df['speaker'].map(df_std['polar-rtheta'])
asl.df['polar-lr-std'] = asl.df['speaker'].map(df_std['polar-lr'])
asl.df['polar-lt-std'] = asl.df['speaker'].map(df_std['polar-ltheta'])

asl.df['norm-polar-rr'] = (asl.df['polar-rr'] - asl.df['polar-rr-mean']) / asl.df['polar-rr-std']
asl.df['norm-polar-rt'] = (asl.df['polar-rtheta'] - asl.df['polar-rt-mean']) / asl.df['polar-rt-std']
asl.df['norm-polar-lr'] = (asl.df['polar-lr'] - asl.df['polar-lr-mean']) / asl.df['polar-lr-std']
asl.df['norm-polar-lt'] = (asl.df['polar-ltheta'] - asl.df['polar-lt-mean']) / asl.df['polar-lt-std']

# TODO define a list named 'features_custom' for building the training set
features_custom = ['norm-polar-rr', 'norm-polar-rt', 'norm-polar-lr','norm-polar-lt']

# features for polar coordinate values where the nose is not the origin
asl.df['polar-rr2']     = np.sqrt(asl.df['right-x']**2 + asl.df['right-y']**2)
asl.df['polar-rtheta2'] = np.arctan2(asl.df['right-x'], asl.df['right-y'])
asl.df['polar-lr2']     = np.sqrt(asl.df['left-x']**2 + asl.df['left-y']**2)
asl.df['polar-ltheta2'] = np.arctan2(asl.df['left-x'], asl.df['left-y'])

features_custom = ['polar-rr2', 'polar-rtheta2', 'polar-lr2', 'polar-ltheta2']

#///// features for polar coordinate values with normalized features_ground
asl.df['grnd-lx-mean']= asl.df['speaker'].map(df_means['grnd-lx'])
asl.df['grnd-ly-mean']= asl.df['speaker'].map(df_means['grnd-ly'])
asl.df['grnd-rx-mean']= asl.df['speaker'].map(df_means['grnd-rx'])
asl.df['grnd-ry-mean']= asl.df['speaker'].map(df_means['grnd-ry'])

asl.df['grnd-lx-std']= asl.df['speaker'].map(df_std['grnd-lx'])
asl.df['grnd-ly-std']= asl.df['speaker'].map(df_std['grnd-ly'])
asl.df['grnd-rx-std']= asl.df['speaker'].map(df_std['grnd-rx'])
asl.df['grnd-ry-std']= asl.df['speaker'].map(df_std['grnd-ry'])

asl.df['norm-grx'] = (asl.df['grnd-rx'] - asl.df['grnd-rx-mean']) / asl.df['grnd-rx-std']
asl.df['norm-gry'] = (asl.df['grnd-ry'] - asl.df['grnd-ry-mean']) / asl.df['grnd-ry-std']
asl.df['norm-glx'] = (asl.df['grnd-lx'] - asl.df['grnd-lx-mean']) / asl.df['grnd-lx-std']
asl.df['norm-gly'] = (asl.df['grnd-ly'] - asl.df['grnd-ly-mean']) / asl.df['grnd-ly-std']

asl.df['polar-rr2']     = np.sqrt(asl.df['norm-grx']**2 + asl.df['norm-gry']**2)
asl.df['polar-rtheta2'] = np.arctan2(asl.df['norm-grx'], asl.df['norm-gry'])
asl.df['polar-lr2']     = np.sqrt(asl.df['norm-glx']**2 + asl.df['norm-gly']**2)
asl.df['polar-ltheta2'] = np.arctan2(asl.df['norm-glx'], asl.df['norm-gly'])

features_custom = ['polar-rr2', 'polar-rtheta2', 'polar-lr2', 'polar-ltheta2']

#///// features_ground & features_polar & features_delta
features_custom = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta','delta-rx', 'delta-ry', 'delta-lx', 'delta-ly','grnd-rx','grnd-ry','grnd-lx','grnd-ly']

#///// features_norm + right + left
asl.df['norm-r'] = asl.df['norm-rx'] + asl.df['norm-ry']
asl.df['norm-l'] = asl.df['norm-lx'] + asl.df['norm-ly']

# TODO define a list named 'features_custom' for building the training set
features_custom = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly','norm-r','norm-l']

#///// features_norm + right + left
asl.df['grnd-r'] = asl.df['grnd-rx'] + asl.df['grnd-ry']
asl.df['grnd-l'] = asl.df['grnd-lx'] + asl.df['grnd-ly']

# TODO define a list named 'features_custom' for building the training set
features_custom = ['grnd-rx', 'grnd-ry', 'grnd-lx','grnd-ly','grnd-r','grnd-l']

#///// rescale
df_min = asl.df.groupby('speaker').min()
df_max = asl.df.groupby('speaker').max()

asl.df['left-x-min']= asl.df['speaker'].map(df_min['left-x'])
asl.df['left-y-min']= asl.df['speaker'].map(df_min['left-y'])
asl.df['right-x-min']= asl.df['speaker'].map(df_min['right-x'])
asl.df['right-y-min']= asl.df['speaker'].map(df_min['right-y'])

asl.df['left-x-max']= asl.df['speaker'].map(df_max['left-x'])
asl.df['left-y-max']= asl.df['speaker'].map(df_max['left-y'])
asl.df['right-x-max']= asl.df['speaker'].map(df_max['right-x'])
asl.df['right-y-max']= asl.df['speaker'].map(df_max['right-y'])

asl.df['rescale-lx'] = (asl.df['left-x'] - asl.df['left-x-min']) / (asl.df['left-x-max'] - asl.df['left-x-min'])
asl.df['rescale-ly'] = (asl.df['left-y'] - asl.df['left-y-min']) / (asl.df['left-y-max'] - asl.df['left-y-min'])
asl.df['rescale-rx'] = (asl.df['right-x'] - asl.df['right-x-min']) / (asl.df['right-x-max'] - asl.df['right-x-min'])
asl.df['rescale-ry'] = (asl.df['right-y'] - asl.df['right-y-min']) / (asl.df['right-y-max'] - asl.df['right-y-min'])

# TODO define a list named 'features_custom' for building the training set
features_custom = ['rescale-rx', 'rescale-ry', 'rescale-lx','rescale-ly']

#///// rescale feature_ground
df_min = asl.df.groupby('speaker').min()
df_max = asl.df.groupby('speaker').max()

asl.df['left-x-min']= asl.df['speaker'].map(df_min['grnd-lx'])
asl.df['left-y-min']= asl.df['speaker'].map(df_min['grnd-ly'])
asl.df['right-x-min']= asl.df['speaker'].map(df_min['grnd-rx'])
asl.df['right-y-min']= asl.df['speaker'].map(df_min['grnd-ry'])

asl.df['left-x-max']= asl.df['speaker'].map(df_max['grnd-lx'])
asl.df['left-y-max']= asl.df['speaker'].map(df_max['grnd-ly'])
asl.df['right-x-max']= asl.df['speaker'].map(df_max['grnd-rx'])
asl.df['right-y-max']= asl.df['speaker'].map(df_max['grnd-ry'])

asl.df['rescale-lx'] = (asl.df['grnd-lx'] - asl.df['left-x-min']) / (asl.df['left-x-max'] - asl.df['left-x-min'])
asl.df['rescale-ly'] = (asl.df['grnd-ly'] - asl.df['left-y-min']) / (asl.df['left-y-max'] - asl.df['left-y-min'])
asl.df['rescale-rx'] = (asl.df['grnd-rx'] - asl.df['right-x-min']) / (asl.df['right-x-max'] - asl.df['right-x-min'])
asl.df['rescale-ry'] = (asl.df['grnd-ry'] - asl.df['right-y-min']) / (asl.df['right-y-max'] - asl.df['right-y-min'])

# TODO define a list named 'features_custom' for building the training set
features_custom = ['rescale-rx', 'rescale-ry', 'rescale-lx','rescale-ly']

#///// rescale features_polar
df_min = asl.df.groupby('speaker').min()
df_max = asl.df.groupby('speaker').max()

asl.df['polar-rr-min']= asl.df['speaker'].map(df_min['polar-rr'])
asl.df['polar-rt-min']= asl.df['speaker'].map(df_min['polar-rtheta'])
asl.df['polar-lr-min']= asl.df['speaker'].map(df_min['polar-lr'])
asl.df['polar-lt-min']= asl.df['speaker'].map(df_min['polar-ltheta'])

asl.df['polar-rr-max']= asl.df['speaker'].map(df_max['polar-rr'])
asl.df['polar-rt-max']= asl.df['speaker'].map(df_max['polar-rtheta'])
asl.df['polar-lr-max']= asl.df['speaker'].map(df_max['polar-lr'])
asl.df['polar-lt-max']= asl.df['speaker'].map(df_max['polar-ltheta'])

asl.df['rescale-polar-rr'] = (asl.df['polar-rr'] - asl.df['polar-rr-min']) / (asl.df['polar-rr-max'] - asl.df['polar-rr-min'])
asl.df['rescale-polar-rt'] = (asl.df['polar-rtheta'] - asl.df['polar-rt-min']) / (asl.df['polar-rt-max'] - asl.df['polar-rt-min'])
asl.df['rescale-polar-lr'] = (asl.df['polar-lr'] - asl.df['polar-lr-min']) / (asl.df['polar-lr-max'] - asl.df['polar-lr-min'])
asl.df['rescale-polar-lt'] = (asl.df['polar-ltheta'] - asl.df['polar-lt-min']) / (asl.df['polar-lt-max'] - asl.df['polar-lt-min'])

# TODO define a list named 'features_custom' for building the training set
features_custom = ['rescale-polar-rr', 'rescale-polar-rt', 'rescale-polar-lr','rescale-polar-lt']
