#import utility as ut
# from particlefilter import particleFilter as pf
import firedrake as fdrake
#import utilitiesFiredrake as utfd
import numpy as np


def load_chk_point(filename, func, n="SSH"):
    with DumbCheckpoint(filename, mode=FILE_READ) as chk:
        chk.load(func, name=n)


def mean_squared_value(x):
    """
    Assume x is an array of 2 dimensional data, we compute

    1/N * \sum_{i=1}{N} \| x[i] \|_{l^2}^2

    or 

    \sum_{i=1}{N} \| x[i] \|_{l^2}^2 * dx * dx
    
    dx = dy = 1/N   since our domain is [0,1]**2

    where the norm inside the sum is the Euclidean 2 norm.
    :param x:
    :return:
    """
    value = np.dot(x[:,0], x[:,0]) + np.dot(x[:,1], x[:,1])
    return np.divide(value, len(x), dtype=np.longdouble)


class FilterDiagnostics(object):
    """
    data handling class to handle all particle filter diagnostics data
    """
    def __init__(self, outputdir):
        # print(outputdir)
        self.outputdir = outputdir
        # self.output_file = fdrake.File(outputdir + 'visuals.pvd')
        # print(outputdir + 'visuals.pvd')

    def generate_diagnostics(self, _filter, _truth, _ess, _weight_exponents, _weight_exponents_uq, _tempering, _step):
    # def generate_diagnostics(self, _filter, _truth, _truth_pv, _ess, _tempering, _obs_err, _step):
        """
        :param _truth: firedrake function of the truth (expect velocity field)

        :param _ess: particle filter ensemble ess, according to the last tempered

        :param _ess_uq: no particle filter ensemble ess

        :param _tempering: particle filter number of tempering steps

        :param _step:
        :return:
        """
        # _filter = pf.Filter()

        ens_mean = _filter.get_ensemble_mean(uq_flag=False, suffix='_step_{}_1'.format(_step))

        ens_mean_prior = _filter.get_prior_ensemble_mean(suffix='_step_{}_1'.format(_step))

        ens_mean_uq = _filter.get_ensemble_mean(uq_flag=True, suffix='_step_{}_1'.format(_step))

        diff = ens_mean - _truth
        diff_uq = ens_mean_uq - _truth
        diff_prior = ens_mean_prior - _truth   # no obs noise

        #print("diff shape " ,diff.shape)

        mse = np.linalg.norm(diff)**2  # no normalisation
        mse_uq = np.linalg.norm(diff_uq)**2  # no normalisation
        mse_prior = np.linalg.norm(diff_prior)**2 


        _ess_non_tempered = _filter.ess_statistic(_filter.normalised_weights(_weight_exponents))
        _ess_uq = _filter.ess_statistic(_filter.normalised_weights(_weight_exponents_uq))

        print("diagnostics", _tempering)

        np.save(self.outputdir+ 'mse_step_{}'.format(_step), [mse])
        np.save(self.outputdir+ 'mse_uq_step_{}'.format(_step), [mse_uq])
        np.save(self.outputdir+ 'mse_prior_step_{}'.format(_step), [mse_prior])     #fdrake mse truth truth -- full domain
        np.save(self.outputdir+ 'ess_step_{}'.format(_step), [_ess])
        np.save(self.outputdir+ 'ess_no_temper_step_{}'.format(_step), [_ess_non_tempered])
        np.save(self.outputdir+ 'ess_uq_step_{}'.format(_step), [_ess_uq])
        np.save(self.outputdir+ 'tempering_step_{}'.format(_step), [_tempering])

    def generate_plot(self, data, data_names, plot_dir, plot_name):
        import matplotlib.pyplot as plt

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=11)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=10)
        
        fig, ax = plt.subplots(dpi=300, figsize=(7,5))
        
        export_dir = plot_dir + '/{}.svg'.format(plot_name)

        legends = []
        
        for dname in data_names:
            line, = ax.plot(data[dname])
            legends.append(line)

        ax.set_xlabel(r'd.a. step')
        #ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
        ax.grid(True, linestyle='--', linewidth=0.5)

        plt.figlegend(legends, data_names) #, loc='center right', prop = {'size':8.5}, frameon=True, borderaxespad=0.1)#, borderaxespad=0.1)
        #plt.subplots_adjust(left = 0.09, top = 0.95, bottom =0.09, right = 0.82, wspace = 0.25, hspace = 0.25)

        plt.tight_layout(pad=1.5)
        plt.savefig(export_dir)
        plt.close()
        #plt.show()

    
    def generate_diagnostics_plots(self, num_steps, data_dir, plot_dir):
        data_names = ['mse', 'mse_uq', 'mse_prior', 'ess', 'ess_uq', 'tempering']

        data={}

        for _name in data_names:
            data[_name] = []

        for _step in range(1, num_steps+1):
            for _name in data_names:
                data[_name].append(np.load(data_dir + '{}_step_{}.npy'.format(_name, _step)))


        mse_names = ['mse', 'mse_uq']
        self.generate_plot({k:data[k] for k in mse_names if k in data}, mse_names, plot_dir, 'mse') 
        ess_names = ['ess', 'ess_uq']
        self.generate_plot({k:data[k] for k in ess_names if k in data}, ess_names, plot_dir, 'ess')
        self.generate_plot({k:data[k] for k in ['tempering'] if k in data}, ['tempering'], plot_dir, 'tempering')

