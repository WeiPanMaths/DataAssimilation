from utility import Workspace

class data_assimilation_workspace(Workspace):
    def __init__(self, workspace_dir):
        Workspace.__init__(self, workspace_dir)
        #self.base_workspace = workspace
        self.observation_data_dir = self.sub_dir('cPDESolution')
        self.ensemble_members_dir = self.sub_dir('EnsembleMembers')
        self.test_files_dir       = self.sub_dir('TestFiles')
        self.param_files_dir      = self.sub_dir('ParamFiles')
        self.fine_resolution_data_dir = self.sub_dir('PDESolution')
        self.particles_dir        = self.sub_dir('Particles')
        self.resampled_particle_dir = self.sub_dir('ResampledParticles')
        self.prior_particle_dir = self.sub_dir('PriorParticles')
        self.diagnostic_dir      = self.sub_dir('Diagnostics')

    def get_diagnostic_dir(self):
        return self.diagnostic_dir

    def get_observation_data_dir(self):
        return self.observation_data_dir

    def get_ensemble_members_dir(self):
        return self.ensemble_members_dir

    def get_test_files_dir(self):
        return self.test_files_dir

    def get_parameter_files_dir(self):
        return self.param_files_dir

    def get_fine_resolution_data_dir(self):
        return self.fine_resolution_data_dir

    def get_particles_dir(self):
        """
        prior 
        """
        return self.particles_dir

    def get_prior_particle_dir(self):
        return self.prior_particle_dir

    def get_resampled_particle_dir(self):
        return self.resampled_particle_dir

