from matplotlib import pyplot as plt


lambdas = {
    'Voting': [1, 1, 10, 0.1, 1],
    'Volcanoes': [100, 100, 100, 100, 100],
    'Spam': [0, 0.1, 0, 0.001, 0.001],
}

m = {
    'Voting': [0, 100, 100, 100, 100],
    'Volcanoes': [1, 0.01, 100, 0, 0.001],
    'Spam': [0.1, 10, 0, 0, 0],
}


plt.style.use('ggplot')


for p, parameter in zip(['lambda', 'm'], [lambdas, m]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1e3)
    datasets = ['Voting', 'Volcanoes', 'Spam']
    ax.boxplot(
        [parameter[dataset] for dataset in datasets],
        labels=datasets,
        showmeans=True,
        showbox=False,
        showfliers=False,
        whis=[0,100],
    )
    fig.savefig(p + 'choices.pdf')
