#   Author: Wei Pan
#   Copyright   2021
#
#   particleFilter.py for TQG
#
#   code for Particle Filter method: vanilla, tempering, nudging, jittering
#

from data_assimilation.relative_import_header import *
import os, errno
import multiprocessing as mp
import numpy as np
from data_assimilation.filterdiagnostics import FilterDiagnostics
from shutil import copyfile
from data_assimilation.data_assimilation_utilities import data_assimilation_workspace
from tqg.example2 import TQGExampleTwo as Example2params
from stqg.solver import STQGSolver
from firedrake_utility import TorusMeshHierarchy
#from firedrake import ensemble, COMM_WORLD
from firedrake import *
#from signal import signal, SIGPIPE, SIG_DFL
#signal(SIGPIPE,SIG_DFL)  # suppress SIGPIPE error


#use_existing_truth_flag = True
#ref_obs_period = 0.02
#twin_ref_obs_period = 0.1

ensemble_comm = ensemble.Ensemble(COMM_WORLD, 1)
spatial_comm = ensemble_comm.comm
# ensemble_comm = ensemble.ensemble_comm


def parallel_run(_func, _args, _num_batches, _nproc=25):
    """
    Parallel execution of _func with arguments _args, in _num_batches batches of _nproc processes

    :param _func:
    :param _args: tuple
    :param _num_batches:
    :param _nproc:
    :return:
    """
    for batch in [[x + _nproc * y for x in range(_nproc)] for y in range(_num_batches)]:
        procs = []
        for job in batch:
            proc = mp.Process(target=_func, args=(job,) + _args)
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

# hard coded for now
class particle_filter_parameters (object):
    def __init__(self, workspace):
        self.number_of_particles = 25
        self.obs_scaling = 100
        #self.obs_period = 1   # multiple of fine resolution dt
        self.obs_period = 0.006  # 3 hours, or 6 saved cPDE files
        self.obs_period_findex_gap = 6
        self.number_of_obs_per_satellite = 50 #900
        self.wkspace = workspace

    def observation_mesh(self, t):
        """
        generates observation mesh that corresponds to two satellite tracks

        Input t is pde saved data file index
             every t is thus 35 minutes.
        Time interval between two consecutively numbered data files  are defined by dumpfreq
        in the solver
        Here we assume the dump_freq = 10        
        and dt = 0.0001 which is roughtly 3.5 minutes
        """
        number_of_track_samples = self.number_of_obs_per_satellite

        r = 1./2./np.pi
        number_of_tracks = 4
        number_of_tracks2 = 5
        total_theta  = 2 * np.pi * number_of_tracks
        total_theta2 = 2 * np.pi * number_of_tracks2

        k_tilde = number_of_tracks/number_of_tracks2
        k_sum = number_of_tracks2 + number_of_tracks
        scaling = number_of_track_samples / k_sum

        ms = np.linspace(0, number_of_track_samples, number_of_track_samples, endpoint=False)
        ns = np.linspace(0, k_sum, k_sum, endpoint=False) 

        ns *= scaling
        ns = np.setdiff1d(ns, np.delete(ns, np.where(np.mod(ns,1)==0))).astype(int)
        ms_2 = np.setdiff1d(ms, ns)

        theta2 = total_theta2 / number_of_track_samples * ms_2
        theta = k_tilde*total_theta2 / number_of_track_samples * ms

        x2= theta2/total_theta2
        x = theta/total_theta #np.zeros(len(y))

        dump_freq = 10
        delta_t = 0.0001
        delta_t_per_save = dump_freq * delta_t  
        delta_t_per_day = 411 * delta_t
        delta_t_per_10_days = 4110 * delta_t


        c = t * 2 * np.pi * delta_t_per_save / delta_t_per_10_days
        
        # satellite 1
        z = r * np.sin(theta + c)
        y = r * np.cos(theta + c)

        # satellite 2
        z2 = r * np.sin(-theta2 + c)
        y2 = r * np.cos(-theta2 + c)

        #intersection points
        #z3 = r * np.sin(intersections * k_tilde + c)
        #y3 = r * np.cos(intersections * k_tilde + c)

        y  = np.arccos(z/r) *  np.sign(y) * r + 0.5
        y_ = np.arccos(z2/r) * np.sign(y2) * r + 0.5
        #y__= np.arccos(z3/r) * np.sign(y3) * r + 0.5

        # now we collect together the two satellite tracks 
        # without any duplicate points
        combine_xs = np.concatenate((x, x2))
        combine_ys = np.concatenate((y, y_))

        combined = np.vstack((combine_xs, combine_ys)).T
        #print(len(x), len(x2), scaling)

        return combined
    
    ### ad hog for now ###
    def reduced_observation_mesh(self, t, wspace):
        # load reduced observation indices
        #wspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")
        sub_obs_site_indices = np.load(wspace.output_name('sub_obs_site_indices.npy', 'ParamFiles'))
        return self.observation_mesh(t)[sub_obs_site_indices]

    #def get_pde_truth_dir():
    #    return output_directory('', '/PDETruths')

    #def get_pde_truth_filename(_t, _dir):
    #    return "{}pde_truth_{}".format(_dir, _t)

    def remove_files(self,dirpath):
        import shutil
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)

    def get_resampled_particle_dir(self):
        return self.wkspace.get_resampled_particle_dir() 

    def get_prior_particle_dir(self):
        #return output_directory('','/ParticleFilter/PriorParticles')
        return self.wkspace.get_prior_particle_dir()

    #def get_initial_truth_filename():
    #    return get_pde_filename(pde_initial_truth_index)

    def get_uq_particle_name(self,_id, _res=64, _var=0.5, suffix=''):
        """
        particle name when no particle filter
        :param _id:
        :param _res:
        :param _var:
        :return:
        """
        #return output_directory('particle_{}{}'.format(_id, suffix), '/uq/SPDE/{}by{}/{}'.format(_res, _res, _var))
        return self.wkspace.output_name('particle_{}{}'.format(_id, suffix), 'UQ/SPDE')

    def get_particle_initial_filename(self, _res, _id):
        #return get_initial_ensemble_dir(_res) + 'particle_{}_0'.format(_id)
        return self.wkspace.get_ensemble_members_dir() + 'ensemble_member_{}'.format(_id)

    def get_particle_name(self,_id, _dir, suffix=''):
        return _dir + "particle_{}{}".format(_id, suffix)


class particle_filter (object):

    def __init__(self, particle_filter_params): #, comm = ensemble_comm):

        self.pf_params = particle_filter_params

        #self.da_wkspace = data_assimilation_workspace(workspace)
        self.da_wkspace = self.pf_params.wkspace

        #_numofparticles, _eof_dir, _var=0.5, _particle_msh_res=64, _obs_msh_res=8, _ess_thresh=0.8, _obs_period=0.1, _obsscaling=10.
        ## filtering parameters #######################################################################
        self.num_of_particles = self.pf_params.number_of_particles
        self.obs_period = self.pf_params.obs_period
        self.obsscaling = self.pf_params.obs_scaling  # artificial scaling for obs variance. Bigger the value, more confidence in observed value
        self.filter_step = 0
        self.uq_pf_flag = True          # false means no assimilation, only UQ
        self.use_existing_truth_flag = True

        ## misc parameters ############################################################################
        self.initial_ensemble_dir = self.da_wkspace.get_ensemble_members_dir() 
        self.particles_dir = self.da_wkspace.get_particles_dir()
        self.resampled_particle_outdir = self.da_wkspace.get_resampled_particle_dir()
        self.prior_particle_outdir =self.da_wkspace.get_prior_particle_dir()
        #self.observationdir = self.da_wkspace.get_observation_data_dir()
        #self.pde_dir = self.da_wkspace.get_pde_dir()
        #self.obs_dumbchkpt_name = "ObsVar"
        #self.obsvarfilename = "obs_var"
        #self.particle_msh_res = _particle_msh_res
        #self.base_t = ut.pde_initial_truth_index

        ## numerical parameters #######################################################################
        cnx = 32
        self.particle_msh = TorusMeshHierarchy(cnx, cnx, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
        self.particle_dt = 0.001 # due to dt * dump_freq used in the calibration process 
        self.particle_msh_res = 32
        self.particle_dump_freq = self.pf_params.obs_period_findex_gap 
        #self.xi_variance = _var
        #self.eof_dir = _eof_dir

        #_obs_grid_xaxis = np.linspace(0, 1, _obs_msh_res + 1)

        self.obs_grid = None # np.array([[x, y] for y in _obs_grid_xaxis for x in _obs_grid_xaxis])

        #obs_var = Function(VectorFunctionSpace(self.particle_msh, "DG", 1))
        #utfd.load_chk_point(self.observationdir + self.obsvarfilename, obs_var, n=self.obs_dumbchkpt_name)
        #self.obs_std_dev = None #self.obsscaling*np.sqrt(np.asarray(obs_var.at(self.obs_grid, tolerance=1e-10)), dtype=np.longdouble)

        self.ess_threshold = 0.8 * self.num_of_particles
        #self.obs_grid_store_indices = ut.get_large_grid_indices(ut.GridUQ.gdxaxis, _obs_grid_xaxis)


    def run_particle_filter_uq_stability(self, _numsteps=5, nproc=25):
        """
        evolve all particles in the ensemble from t_i to t_{i+1}
        compute ess
        resample (if required)

        tempering step added

        before tempering, particles are evolved and stored to self.resampled_particle_outdir
        after tempering, particles are resampled and mcmc'ed, and stored to self.particles_dir

        :param _numsteps: number of d.a. steps, should match up with assimilation lead time
        :return:
        """
        # clear data ######################################################################################
        #if not self.use_existing_truth_flag:
        #    print("clearing data ... ")
        #    if twin_experiment_flag:
        #        ut.remove_files(ut.get_spde_truth_dir(_res=self.particle_msh_res, _var=self.xi_variance)[:-1])
        #    else:
        #        ut.remove_files(ut.get_pde_truth_dir()[:-1])
        #ut.remove_files(self.particles_dir[:-1])
        #ut.remove_files(self.prior_particle_outdir[:-1])
        #ut.remove_files(ut.output_directory('', '/uq/SPDE/{}by{}/{}'.format(self.particle_msh_res, self.particle_msh_res, self.xi_variance))[:-1])
        ###################################################################################################

        assert self.num_of_particles >= nproc

        #print('start experiment: number particles {}, observation period {}, number d.a. steps {}, ' 'total period {} ett, obsscaling {}'.format(self.num_of_particles, self.obs_period, _numsteps, _numsteps * self.obs_period / utfd.eddy_turnover_time, self.obsscaling))
        print('start experiment')
        _num_batches = int(round(self.num_of_particles / nproc))
        print(_num_batches)

        #ut.remove_files(diagnosticsdir[:-1])
        diagnostics = FilterDiagnostics(outputdir=self.da_wkspace.diagnostic_dir)

        # multiprocessing storage data
        manager = mp.Manager()
        weights_exponents = manager.Array('d', range(self.num_of_particles))
        weights_exponents_uq = manager.Array('d', range(self.num_of_particles))
        random_state_store = manager.dict()  # needed by mcmc #

        def my_wrapper(_id, _weights, bm_paths_store, _obs):
            data_field, bm_path = self.evolve_particle(_id)
            bm_paths_store[_id] = bm_path
            signal = np.asarray(data_field.at(self.obs_grid, tolerance=1e-10))
            _weights[_id] = self.compute_log_likelihood(signal.reshape(len(signal),1), _obs)

        def my_wrapper_nopf(_id, _weights, _obs):
            data_field = self.evolve_particle_nopf(_id)
            signal = np.asarray(data_field.at(self.obs_grid, tolerance=1e-10))
            _weights[_id] = self.compute_log_likelihood(signal.reshape(len(signal),1), _obs)

        import time
        
        for _step in range(_numsteps):
            
            # update observation grid 
            # _t is for observation_grid
            _t = (self.filter_step + 1)*self.pf_params.obs_period_findex_gap
            self.obs_grid = self.pf_params.reduced_observation_mesh(_t, self.da_wkspace)
            
            print("\n----- d.a. step: {} / {} ".format(self.filter_step+1, _numsteps), " -----------")

            # _obs is _truth.at + _obs_err
            #_obs, _truth, _obs_err, _truth_pv = obs_func()
            _obs, _truth = self.generate_observation(_t)

            # run particle forward and compute results weights in parallel
            # the weights vector store the log likelihoods given observations

            start = time.perf_counter()

            parallel_run(my_wrapper, (weights_exponents, random_state_store, _obs), _num_batches, _nproc=nproc)
            parallel_run(my_wrapper_nopf, (weights_exponents_uq, _obs), _num_batches,_nproc=nproc)
            diff = time.perf_counter() - start
            print(weights_exponents)

            #copy particles to prior distribution folder
            for _id in range(self.num_of_particles):
                copyfile(self.particle_name(_id, self.resampled_particle_outdir)+'_1.h5', 
                        self.particle_name(_id, self.prior_particle_outdir, suffix='_step_{}_1'.format(self.filter_step+1)+'.h5'))

            # apply tempering and jittering and obtain the final ess value
            start = time.perf_counter()
            ess, numtemperingsteps = self.tempering_step(weights_exponents, random_state_store, _obs, manager)
            diff += time.perf_counter() - start

            # print("time: ", diff)
            self.filter_step += 1

            # run diagonstics here to save information about the posterior distribution
            diagnostics.generate_diagnostics(self, _truth, ess, weights_exponents, weights_exponents_uq, numtemperingsteps, self.filter_step)
            random_state_store.clear()

        diagnostics.generate_diagnostics_plots(_numsteps, self.da_wkspace.get_diagnostic_dir(), self.da_wkspace.get_test_files_dir())


    def tempering_step(self, weight_exponents, random_store, obs, manager):
        """
        For the tempering step algorithm see writing.

        phi is the temperature, starts at 1 which means no tempering step
        the size of phi is found by bisection

        :param obs: observation (truth + obs noise)
        :param weight_exponents: log likelihood, maybe modified by parallel execution in resample_mt
        :param random_store: bm paths that were used in the generation of the parent particles
        :return:
        """
        parent_id_dict = manager.dict()  # key:value key is particle id, and value is the parent particle id

        for i in range(self.num_of_particles):
            parent_id_dict[i] = i

        remaining_temperature_step = 1.   # temperature remaining = len( (previous, 1] )

        num_tempering_steps = 0

        if self.uq_pf_flag:
            while not self.resampling_test_statistic(weight_exponents, remaining_temperature_step):
                # find the next temperature \in (1-remaining_temperature_step, 1]
                # this step doesn't change weight_exponents
                new_phi = self.get_temperature(1.-remaining_temperature_step, weight_exponents)
                temp_inc = new_phi + remaining_temperature_step - 1.  # = phi(r) - phi(r-1) in paper

                #print("tempering step ", num_tempering_steps, " new temp: ", new_phi)

                # resample + MCMC to obtain a new set of particles
                # new set of particles gives different likelihoods, so weight_exponents are updated
                self.resample_mt(weight_exponents, random_store, obs, new_phi, temp_inc, parent_id_dict, manager)
                # np.savetxt(output_directory('acceptance_rates_pfStep{}_tempStep{}.csv'.format(self.filter_step, num_tempering_steps),
                #                          "/TestResults/MCMC"), accept_rate, delimiter=',')
                remaining_temperature_step = 1. - new_phi
                # print("tempering step {}, remaining temp {} -----------------------------".format(num_tempering_steps, remaining_temperature_step))
                num_tempering_steps += 1

            # make the last step then we don't need equal weights
            # print("tempering step (last): {} -------------------------------------".format(num_tempering_steps))
            print("last tempering step ", num_tempering_steps)
            self.resample_mt(weight_exponents, random_store, obs, 1., remaining_temperature_step, parent_id_dict, manager)
            #print("after resample_mt")
            # np.savetxt(
            #     output_directory('acceptance_rates_pfStep{}_tempStep{}.csv'.format(self.filter_step, num_tempering_steps),
            #                      "/TestResults/MCMC"), accept_rate, delimiter=',')
            print("\n")

        # final_ess = self.ess_statistic(self.normalised_weights(weight_exponents))
        final_ess = self.ess_statistic(self.normalised_weights(weight_exponents, remaining_temperature_step))    # this should include remaining_temperature_step
        print("final ess: {}, no temper ess: {}".format(final_ess, self.ess_statistic(self.normalised_weights(weight_exponents)))) 

        # resave particles to /Particles/ for the next step
        for _id in range(self.num_of_particles):
            newname = self.particle_name(_id, self.particles_dir, suffix='_step_{}_1'.format(self.filter_step+1))
            #print('tempering ', newname)
            self.rename_pv(self.particle_name(_id, self.resampled_particle_outdir, suffix='_1'), newname)

        parent_id_dict.clear()

        # clear resampled_particle_outdir
        self.pf_params.remove_files(self.pf_params.get_resampled_particle_dir()[:-1])

        return final_ess, num_tempering_steps


    def evolve_particle(self, parent_id, **kwargs):
        """
        run particle of id 'id' forward until the next observation time

        :param parent_id: parent particle id, if no child id supplied

        :kwargs random_state: Random number seed for recovering the original Brownian Motion path for use in the MCMC move
        :kwargs proposal_step: Rho in the MCMC move
        :kwargs state_store: Data store for storing the random number seed used for generating the BM path. This data store covers the whole ensemble thus works with multiprocessing.
        :kwargs resampling_output_id: new id when the method is called by mcmc jittering, rename the moved particle

        :return: velocity field and the bm path used by the solution operator. the latter is needed in mcmc
        """
        #pv_input = Function(FunctionSpace(self.particle_msh, "DG", 1))
        pname = self.particle_initial_condition_name(parent_id, suffix='_step_{}_1'.format(self.filter_step))

        #print('evolve ', pname)

        #tutfd.load_chk_point(pname, pv_input, "Vorticity")

        # evolve particle
        angry_dolphin_params = Example2params(self.pf_params.obs_period, self.particle_dt, self.particle_msh, bc='x', alpha=None)
        #p_i = particle.Particle(self.obs_period, self.particle_dt, self.particle_msh, parent_id, self.eof_dir, self.xi_variance)
        p_i = STQGSolver(angry_dolphin_params) 
        p_i.id = parent_id
        output_id = kwargs.get('resampling_output_id')

        # if resampling, create a temp particle name
        if 'resampling_output_id' in kwargs:
            p_i.id = output_id
            _outputname = self.temp_particle_name(output_id, self.resampled_particle_outdir)
        else:
            _outputname = self.particle_name(parent_id, self.resampled_particle_outdir)
            # _outputname = self.particle_name(parent_id, self.prior_particle_outdir)

        # output name does not have step suffix attached
        #bm_path = p_i.spdesolver(pv_input, _outputname, True, **kwargs)
        p_i.load_initial_conditions_from_file(pname)

        zeta_fname = self.da_wkspace.output_name("zetas.npy", "ParamFiles")
        bm_path = p_i.solve(self.particle_dump_freq, _outputname, _outputname, ensemble_comm, do_save_data=True, do_save_visual=False, do_save_spectrum=False, res=32,  zetas_file_name=zeta_fname)

        # get corresponding velocity on obs_msh
        return p_i.ssh, bm_path


    def evolve_particle_nopf(self, parent_id, **kwargs):
        """
        run particle of id 'id' forward until the next observation time

        :param parent_id: parent particle id, if no child id supplied

        :kwargs random_state: Random number seed for recovering the original Brownian Motion path for use in the MCMC move
        :kwargs proposal_step: Rho in the MCMC move
        :kwargs state_store: Data store for storing the random number seed used for generating the BM path. This data store covers the whole ensemble thus works with multiprocessing.
        :kwargs resampling_output_id: new id when the method is called by mcmc jittering, rename the moved particle

        :return: velocity field and the bm path used by the solution operator. the latter is needed in mcmc
        """

        pname = self.get_uq_particle_name(parent_id, suffix='_step_{}_1'.format(self.filter_step))

        # evolve particle
        #p_i = particle.Particle(self.obs_period, self.particle_dt, self.particle_msh, parent_id, self.eof_dir, self.xi_variance)
        angry_dolphin_params = Example2params(self.pf_params.obs_period, self.particle_dt, self.particle_msh, bc='x', alpha=None)
        #p_i = particle.Particle(self.obs_period, self.particle_dt, self.particle_msh, parent_id, self.eof_dir, self.xi_variance)
        p_i = STQGSolver(angry_dolphin_params) 
        p_i.id = parent_id

        _outputname = self.pf_params.get_uq_particle_name(parent_id, suffix='_step_{}'.format(self.filter_step+1))
        p_i.load_initial_conditions_from_file(pname)

        zeta_fname = self.da_wkspace.output_name("zetas.npy", "ParamFiles")

        p_i.solve(self.particle_dump_freq, _outputname, _outputname, ensemble_comm, do_save_data=True, do_save_visual=False, do_save_spectrum=False, res=32, zetas_file_name=zeta_fname)

        return p_i.ssh


    def mcmcjitter(self, parent_id, child_id, parent_bm_path, parent_loglikelihood, obs, temperature_k):
        """
        Metropolis-Hastings

        :param child_id:            new id for storing the newly generated particles
               parent_bm_path:      parent bm paths
               old_id:              chosen particle id for getting the original bm path
               parent_loglikelihood:    for computing the accept/reject probability;
               obs:                 observation for computing weight exponents
               temperature_k:    k'th temperature, see algorithm 3 in paper
        :return: new weight exponent, and bm path for the accepted particle
        """
        # def my_wrapper(_id, _weights, _state, _obs):
        #     signal = np.asarray(self.evolve_particle(_id).at(self.obs_grid, tolerance=1e-10))
        #     self.compute_weight_exponent(_weights, _id, signal, _obs)
        rho = 0.9

        np.random.seed(None)
        _new_loglikelihood = 0.

        # accept = False
        # bm_path = None
        # while not accept:
        #     data_field, bm_path = self.evolve_particle(parent_id, resampling_output_id=child_id, proposal_step=rho, state_store=parent_bm_path)
        #     x = np.asarray(data_field.at(self.obs_grid, tolerance=1e-10))
        #     #make copies, so we dont write over the old values
        #     new_weight_exponents = np.array(weight_exponents)
        #     old_weight_exponents = np.array(weight_exponents)
        #     _new_weight_exponent = self.compute_weight_exponent(x, obs)
        #     # replace the parent particle weight to compare likelihood, hence here it should be old_id
        #     new_weight_exponents[parent_id] = _new_weight_exponent
        #     numerator = self.normalised_weights(new_weight_exponents, temperature)[parent_id]
        #     denominator = self.normalised_weights(old_weight_exponents, temperature)[parent_id]
        #     prob = min(1., numerator / denominator)
        #     alpha = np.random.uniform()
        #     accept = True if alpha < prob else False
        # accept_count = 0.
        size = 5

        mh_bm_path = np.array(parent_bm_path)  # create a copy
        _old_log_likelihood = parent_loglikelihood  # i.e. self.compute_log_likelihood(parent, obs)

        for i in range(size):
            data_field, bm_path = self.evolve_particle(parent_id, resampling_output_id=child_id, proposal_step=rho, state_store=mh_bm_path)
            x = np.asarray(data_field.at(self.obs_grid, tolerance=1e-10))
            

            # make copies, so we dont write over the old values
            # new_weight_exponents = np.array(weight_exponents)
            # old_weight_exponents = np.array(weight_exponents)

            _new_loglikelihood = self.compute_log_likelihood(x.reshape(len(x), 1), obs)  # value

            # replace the parent particle weight to compare likelihood, hence here it should be old_id
            # new_weight_exponents[parent_id] = _new_weight_exponent

            # numerator = self.normalised_weights(new_weight_exponents, temperature)[parent_id]
            # denominator = self.normalised_weights(old_weight_exponents, temperature)[parent_id]

            # prob = min(1., numerator / denominator)
            likelihoodratio = np.exp(temperature_k * (_new_loglikelihood - _old_log_likelihood))

            prob = min(1., likelihoodratio)
            alpha = np.random.uniform()

            if alpha < prob:
                # accept
                mh_bm_path = np.array(bm_path)
                _old_log_likelihood = _new_loglikelihood
                # accept_count += 1.0

        return _new_loglikelihood, mh_bm_path  # bm_path

    def resample_mt(self, weight_exponents, random_store, obs, temp_k, temp_inc, parent_id_dict, manager):
        """
        (weight_exponents, random_store, obs, new_phi, temp_inc, parent_id_dict, manager)

        multithreaded version of resampling

        randomly draw new samples based on the current normalised weights
        apply mcmc to the repeated samples
        update the weight_exponents

        :param weight_exponents: current weight exponents
        :param random_store: bm path storage
        :param obs: used to update weight exponent after mcmc generated a new particle
        :param temp_k: new_phi
        :param temp_inc: temperature increment for resampling # phi(r) - phi(r-1)
        :param parent_id_dict: dictionary where key:value is particle_id: parent_particle_id
        :return:
        """
        def mcmc_thread(_parent_id, _new_id, _weight_exponents, _obs, _temp_k, weight_exponents_save, _bm_path,
                        random_store_save, parent_id_dict_save):
            """
            :param _parent_id:
            :param _new_id:
            :param _weight_exponents: doesn't get written over
            :param _obs:
            :param _temp_k: new phi 
            :param weight_exponents_save: weight_exponents_cp
            :param _bm_path:
            :param random_store_save:
            :param parent_id_dict_save:
            :return:
            """
            p_loglikelihood = _weight_exponents[_parent_id]
            _temp_exponent, new_bm_path = self.mcmcjitter(_parent_id, _new_id, _bm_path, p_loglikelihood, _obs, _temp_k)
            weight_exponents_save[_new_id] = _temp_exponent
            random_store_save[_new_id] = new_bm_path
            parent_id_dict_save[_new_id] = _parent_id

        #print("start resampling")
        weight_exponents_cp = mp.Array('d', weight_exponents)  # copy weight_exponent
        random_store_cp = manager.dict(random_store)  # copy random store
        parent_id_dict_cp = manager.dict(parent_id_dict)  # copy parent_id_dict

        # base case: weight_exponents ~ pi(0) = p(t) or zeroth tempered distribution, temp remaining = 1.
        # induction step: assume weight_exponets ~ pi(k-1), we want weight_exponents ~ pi(k)
        # resample_indices_counts resamples according to the k'th tempered distribution
        # hence require the temperature increment
        # ids_counts contains the resampled particle ids, and how many copies
        # total mcmc steps is sum of all copies with size > 1, i.e. how many mcmc jittering we need to do
        ids_counts, total_mcmc_steps = self.resample_indices_counts(weight_exponents, temp_inc)
        resampled_ids, counts = zip(*ids_counts)
        resampled_ids = np.asarray(resampled_ids)
        counts = np.asarray(counts)

        # print("total mcmc steps ", total_mcmc_steps)
        # acceptance_rates_array = mp.Array('f', np.zeros(total_mcmc_steps))  # of a single tempering step
        # thread_id = 0

        nprocs = 25
        procs = []
        remaining_steps = int(total_mcmc_steps)  # make a copy of the value

        new_particle_id = 0
        # need to re-zip because as an iterator ids_counts is empty
        for resampled_id, count in zip(resampled_ids, counts):
            parent_id = parent_id_dict[resampled_id]
            child_id_range = np.arange(new_particle_id, new_particle_id + count, 1)

            if count > 1:
                # only jitter the copies, hence this separation of indices
                for new_id in child_id_range[:1]:
                    #print('in loop', self.get_resampled_particle_name(resampled_id))
                    self.rename_pv(self.get_resampled_particle_name(resampled_id),   # get evolved particle name in resampled_particle_outdir
                                   self.temp_particle_name(new_id, self.resampled_particle_outdir, '_1'))
                    # exponents don't change. tempered distribution is reflected in resample_indices_counts
                    weight_exponents_cp[new_id] = weight_exponents[resampled_id]
                    # bm trajectory doesn't change
                    random_store_cp[new_id] = random_store[resampled_id]
                    # parent doesn't change
                    parent_id_dict_cp[new_id] = parent_id

                # only jitter duplicates
                for new_id in child_id_range[1:]:
                    proc = mp.Process(target=mcmc_thread, args=(parent_id, new_id, weight_exponents, obs, temp_k, weight_exponents_cp,
                                                                random_store[resampled_id], random_store_cp, parent_id_dict_cp))
                    procs.append(proc)
                    proc.start()

                    batch_len = min(remaining_steps, nprocs)  # multithread_batch_length(remaining_steps, nprocs)
                    if len(procs) == batch_len:
                        for proc in procs:
                            proc.join()
                        procs.clear()
                        remaining_steps -= batch_len
            new_particle_id += count

        new_particle_id = 0
        for resampled_id, count in zip(resampled_ids, counts):
            parent_id = parent_id_dict[resampled_id]
            child_id_range = np.arange(new_particle_id, new_particle_id + count, 1)
            if count == 1:
                new_id = child_id_range[0]
                self.rename_pv(self.get_resampled_particle_name(resampled_id),
                               self.temp_particle_name(new_id, self.resampled_particle_outdir, '_1'))
                # exponents don't change. tempered distribution is reflected in resample_indices_counts
                weight_exponents_cp[new_id] = weight_exponents[resampled_id]
                # bm trajectory doesn't change
                random_store_cp[new_id] = random_store[resampled_id]
                # parent doesn't change
                parent_id_dict_cp[new_id] = parent_id
            new_particle_id += count

        if new_particle_id != self.num_of_particles:
            raise Exception('new_particle_id is not number of particles!')

        # for _id in range(new_particle_id):
        for _id in range(self.num_of_particles):
            self.rename_pv(self.temp_particle_name(_id, self.resampled_particle_outdir, '_1'),self.particle_name(_id, self.resampled_particle_outdir, suffix='_1'))  # need _1 here
            weight_exponents[_id] = weight_exponents_cp[_id]
            random_store[_id] = random_store_cp[_id]
            parent_id_dict[_id] = parent_id_dict_cp[_id]

        # return np.asarray(acceptance_rates_array)

    ## observation #####################################################################################################

    def generate_observation(self, t):
        """
        t here corresponds to saved file index,
        which is dumpfreq (=10) * dt (=0.0001)

        """
        # load covariance for observation error
        cov = np.load(self.da_wkspace.get_parameter_files_dir() + 'obs_cov_sub_matrix.npy')
        L = np.linalg.cholesky(cov)
        z_ = np.random.normal(size=(len(L), 1))
        z  = L.dot(z_)

        truth = np.load(self.da_wkspace.get_observation_data_dir() + 'obs_data_reduced.npz')['obs_data_{}'.format(t)]
        truth = truth.reshape((truth.shape[0], 1))
        #print(truth.files[:3])
        #print(truth.shape, z.shape)
        return truth + z, truth


    # misc ############################################################################################################
    def resampling_test_statistic(self, weight_exponents, temp_increment=1.):
        """
        ess test statistic

        can be something else

        Does not update weight_exponents here, nor self.nweights

        :param weight_exponents:
        :param temp_increment:
        :return: true if ess threshold is attained
        """
        _temp_ess = self.ess_statistic(self.normalised_weights(weight_exponents, temp_increment))
        #print("temp inc: {}, ess {}".format(temp_increment, _temp_ess))

        return _temp_ess > self.ess_threshold - 1.e-10

    def resample_indices_counts(self, exponents, temp_increment):
        """
        draw a new set of particles given the current normalised particle weights

        return the drawn particles, represented by their particle id, and how many times the particles were drawn

        :param exponents:
        :param temp_increment:
        :return: list((a,b)) where a is the id of a resampled particle, b is number of times it got picked
        """
        pool = np.cumsum(self.normalised_weights(exponents, temp_increment))
        draws = np.random.uniform(0, 1, pool.shape)
        count = np.zeros(self.num_of_particles, dtype=np.int16)
        for draw in draws:
            count[np.min(np.where(pool > draw))] += 1

        _temp = np.where(count > 0)[0]  # number of distinct
        _temp2 = np.where(count == 1)[0]
        # _num_mcmc_steps = int(np.sum(count[count>1]))   # UNCOMMENT THIS TO JITTER ALL DUPLICATED PARTICLES

        _num_mcmc_steps = self.num_of_particles - _temp.shape[0]  #- _temp2.shape[0]
        #print("resampling: number of resampled particles {}, num of non duplicates {} ".format(_temp.shape[0], _temp2.shape[0]))
        #print("      number of jittering steps ", _num_mcmc_steps)

        return zip(_temp, count[count>0]), _num_mcmc_steps


    def compute_log_likelihood(self, x, y):
        """
        compute -0.5 * \sum \|x[i] - y[i]\|^2 / stddev^2

        :param x:
        :param y:
        :return:
        """
        #ndiff = (x - y)/self.obs_std_dev
        #myvalue = -0.5 * (np.dot(ndiff[:, 0], ndiff[:, 0]) + np.dot(ndiff[:, 1], ndiff[:, 1]))
        #return myvalue
        cov_inv = np.load(self.da_wkspace.get_parameter_files_dir() + 'obs_cov_sub_matrix_inv.npy')
        diff = x - y

        myvalue = -0.5 * diff.T.dot(cov_inv).dot(diff) * self.obsscaling
        #print(myvalue.shape, diff.shape, x.shape, y.shape)
        
        return myvalue


    @staticmethod
    def observe_ssh(cmesh, filename, observation_points):
        #cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
        cssh = Function(FunctionSpace(cmesh, "CG", 1))
        with DumbCheckpoint(filename, mode=FILE_READ) as chk:
            chk.load(cssh, name="SSH")
        values = np.asarray(cssh.at(observation_points, tolerance=1e-10))
        #print(values.shape)
        return values.reshape((len(values),1))


    @staticmethod
    def normalised_weights(_exponents, temp_increment=1.):
        """
        temp_increment < 1 means increased variance
        """
        z = np.asarray(_exponents)
        z -= np.max(z)
        weights = np.exp(z * temp_increment)
        return weights / np.sum(weights)

    def rename_pv(self, old_name, new_name):
        """
        copy n paste a firedrake h5 pv file to have a new name
        :param old_name:
        :param new_name:
        :return:
        """
        #print("rename_pv", old_name, new_name)
        pv  = Function(FunctionSpace(self.particle_msh, "DG", 1))
        psi = Function(FunctionSpace(self.particle_msh, "CG", 1))
        ssh = Function(FunctionSpace(self.particle_msh, "CG", 1))
        b   = Function(FunctionSpace(self.particle_msh, "DG", 1))

        #utfd.load_chk_point(old_name, pv, "Vorticity")
        with DumbCheckpoint(old_name, mode=FILE_READ) as chk:
            chk.load(pv, name="PotentialVorticity")
            chk.load(b, name="Buoyancy")
            chk.load(psi, name="Streamfunction")
            chk.load(ssh, name="SSH")
                
        with DumbCheckpoint(new_name, mode=FILE_CREATE) as chk:
            chk.store(pv, name="PotentialVorticity")
            chk.store(b,  name="Buoyancy")
            chk.store(psi,name="Streamfunction")
            chk.store(ssh,name="SSH")

    def particle_initial_condition_name(self, _id, suffix=''):
        """
        get particle initial condition, i.e. value at t_{i-1}, for the interval (t_{i-1}, t_i]
        :param _id:
        :return:
        """
        # _particle_name = self.initial_ensemble_dir + 'particle_{}_0'.format(_id)
        #_particle_name = ut.get_particle_initial_filename(self.particle_msh_res, _id)
        _particle_name = self.pf_params.get_particle_initial_filename(self.particle_msh_res, _id)
        return _particle_name if self.filter_step == 0 else self.particle_name(_id, self.particles_dir, suffix)
        #self.particles_dir + "particle_{}".format(_id)

    def get_resampled_particle_name(self, _id):
        """
        get evolved particle name
        :param _id:
        :return:
        """
        # return self.particle_name(id, self.resampled_particle_outdir)
        # return self.resampled_particle_outdir + "particle_{}".format(_id)
        _t =  self.pf_params.get_particle_name(_id, self.resampled_particle_outdir, suffix='_1')
        #print("get_resampled_particle name", _t)
        return _t

    def get_prior_particle_name(self, _id, suffix=''):
        return self.particle_name(_id, self.prior_particle_outdir, suffix)

    def get_uq_particle_name(self, parent_id, suffix=''):
        return self.pf_params.get_particle_initial_filename(self.particle_msh_res, parent_id) if self.filter_step == 0 else \
            self.pf_params.get_uq_particle_name(parent_id, suffix=suffix)

    def particle_name(self, _id, _dir, suffix=''):
        # return _dir + "particle_{}".format(_id)
        return self.pf_params.get_particle_name(_id, _dir, suffix)

    def temp_particle_name(self, _id, _dir, suffix=''):
        return _dir + "temp_{}{}".format(_id, suffix)

    def get_temperature(self, low_temp, weight_exponents):
        """
        bisection method for finding the next temperature between current_temperature and 1. such that
        ess > threshold

        :param weight_exponents:
        :param low_temp:
        :return: new temperature
        """
        low = low_temp
        high = 1.
        success = False
        phi = 0.   # initialise phi
        while not success:
            phi = 0.5 * (low + high)
            success = self.resampling_test_statistic(weight_exponents, phi-low)
            high = phi
        return phi

    @staticmethod
    def ess_statistic(weights):
        """
        :param weights: normalised weights
        :return:
        """
        # return 1./ np.sum(np.square(weights))
        return 1. / np.dot(weights, weights)

    def get_ensemble_mean(self, uq_flag=False, suffix=''):
        """ get enemble mean at self.filter_step,   pv here means ssh"""
        funcspace = FunctionSpace(self.particle_msh, "CG", 1)
        ssh_mean = Function(funcspace)
        ssh_input = Function(funcspace)
        ssh_mean.assign(0)

        particle_name = self.get_uq_particle_name if uq_flag else self.particle_initial_condition_name

        for _id in range(self.num_of_particles):
            #utfd.load_chk_point(particle_name(id, suffix=suffix), ssh_input, "SSH")
            _name = particle_name(_id, suffix=suffix)
            #print(_name)
            with DumbCheckpoint(_name, mode=FILE_READ) as chk:
                chk.load(ssh_input, name='SSH')
            ssh_mean += ssh_input

        ssh_mean *= 1. / self.num_of_particles
        #solv = particle.Particle(self.obs_period, self.particle_dt, self.particle_msh, 0, self.eof_dir, self.xi_variance)
        #v = Function(VectorFunctionSpace(self.particle_msh, "DG", 1))
        #v.project(solv.v_given_q(pv_mean))

        #return v, pv_mean
        _mean = np.asarray(ssh_mean.at(self.obs_grid, tolerance=1e-10)) 
        #print("_mean shape", _mean.shape, self.obs_grid.shape)

        return _mean.reshape((_mean.shape[0],1))

    def get_prior_ensemble_mean(self, suffix=''):
        funcspace = FunctionSpace(self.particle_msh, "CG", 1)
        ssh_mean = Function(funcspace)
        ssh_input = Function(funcspace)
        ssh_mean.assign(0)

        for _id in range(self.num_of_particles):
            with DumbCheckpoint( self.particle_name(_id, self.prior_particle_outdir, suffix), mode=FILE_READ) as chk:
                chk.load(ssh_input, name="SSH")
            ssh_mean += ssh_input

        ssh_mean *= 1. / self.num_of_particles
        #solv = particle.Particle(self.obs_period, self.particle_dt, self.particle_msh, 0, self.eof_dir, self.xi_variance)
        #v = Function(VectorFunctionSpace(self.particle_msh, "DG", 1))
        #v.project(solv.v_given_q(pv_mean))

        #return v, pv_mean
        _mean = np.asarray(ssh_mean.at(self.obs_grid, tolerance=1e-10))

        return _mean.reshape((_mean.shape[0],1))
