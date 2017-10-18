function startup()

    addpath(genpath('train'));
    addpath(genpath('eval'));
    addpath(genpath('experiments'));
    addpath(genpath('preprocessing'));

    % First you need to set the path of mosek and cvx (used for optimization)
    libs_path       = '/local/jpeyre/tools';
    mosek_path      = fullfile(libs_path,'mosek/7/toolbox/r2009b');
    mosek_license   = fullfile(libs_path,'/mosek/mosek.lic');
    cvx_path        = fullfile(libs_path,'cvx/cvx_setup.m');
    vl_path         = fullfile(libs_path,'vlfeat-0.9.20/toolbox');

    addpath(mosek_path);
    addpath(vl_path);
    setenv('MOSEKLM_LICENSE_FILE', mosek_license);
    run(cvx_path);
    vl_setup();

    
end
