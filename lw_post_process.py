#!/usr/bin/env python
import numpy as np

################################################
###   File Reading/Writing Utilities   #########
################################################

# Typical column names that aren't sampling parameters
non_params = ["logpost", "deltalogl", "logl", "cycle", "logprior", "loglh1", "logll1", "loglv1",
              "timestamp", "snrh1", "snrl1", "snrv1", "snr", "time_mean", "time_maxl", "t0",
              "phase_maxl", "cosalpha", "azimuth", "evidence_ratio", "acceptance_rate", "walker"]

def pass_header(inp):
    """
    Get past the header of a PTMCMC output file.
    """
    header = inp.readline().split()
    while True:
        if len(header) > 0 and header[0].decode('utf-8') == 'cycle':
            break
        header = inp.readline().split()
    inp.readline()
    return [p.lower().decode('utf-8') for p in header]

def extract_command_line(infile):
    """
    Extract the command line from the header of the PTMCMC output.
    """
    with open(infile, 'r') as inp:
        # Find the command line
        line = inp.readline().split()
        while len(line) < 1 or line[0] != "Command":
            line = inp.readline().split()
    return line

def extract_event_number(infile, event=None, approx=None):
    """
    Figure out the event number being injected
    from the command line in the header of a
    PTMCMC output file.
    """
    cmd = extract_command_line(infile)

    # Figure out the event number
    return int(cmd[cmd.index('--event')+1])

def check_if_0noise(infile):
    cmd = extract_command_line(infile)
    if "--0noise" in cmd:
        return True
    else:
        return False

def generate_default_output_filename(approx=None, event=None):
    import os

    uname = os.getlogin()
    outname = '_'

    if approx is not None:
        outname += '{}_'.format(approx)
    if event is not None:
        outname += '{}_'.format(event)
    outname += '{}'.format(uname)
    outname = '{}.png'.format(outname.strip('_'))
    return outname

def estimate_logl_normalization(infile):
    """
    Roughly estimate the normalization that
    should be removed from the loglikelihood values
    when using marginalized likelihoods.
    """
    time_prior_width = 0.200 # default is 200 ms
    time_post_width = 0.001  # ~1 ms
    phase_prior_width = 2.0 * np.pi
    phase_post_width = 0.2 * phase_prior_width # Assume ~1/5 of prior

    with open(infile, 'r') as inp:
        inp.readline()
        cmdline = inp.readline()
    if cmdline.find('margtimephi'):
        logl_norm = np.log(time_post_width/time_prior_width)
        logl_norm += np.log(phase_post_width/phase_prior_width)
    elif cmdline.find('margtime'):
        logl_norm = np.log(time_post_width/time_prior_width)
    elif cmdline.find('margphi'):
        logl_norm = np.log(phase_post_width/phase_prior_width)
    else:
        logl_norm = 0
    return logl_norm

def get_logl(infile):
    """
    Extract the log-likelihood series of the chain.
    """
    with open(infile, 'rb') as inp:
        header = pass_header(inp)
        try:
            logl_col = header.index('deltalogl')
        except ValueError:
            logl_col = header.index('logl')
        logl = np.genfromtxt(inp, usecols=(logl_col))
    return logl

def get_network_snr(infile):
    """
    Find the injected network SNR in the PTMCMC header.
    """
    with open(infile, 'r') as inp:
        header = inp.readline().split()
        while True:
            if len(header) > 0 and header[0] == 'Detector':
                break
            header = inp.readline().split()

        SNR_col = header.index('SNR')
        single_ifo_snrs = []
        while True:
            try:
                single_ifo_snrs.append(float(inp.readline().split()[SNR_col]))
            except IndexError:
                break

    single_ifo_snrs = np.array(single_ifo_snrs)
    network_snr = np.sqrt(sum(single_ifo_snrs*single_ifo_snrs))
    return network_snr


def get_approx(infile):
    """
    Find the waveform family in the PTMCMC header.
    """
    import lalsimulation as lalsim

    with open(infile, 'r') as inp:
        # Column headers sometimes have a single space in the name
        line = inp.readline()
        sep = '\t' if '\t' in line else '  '
        header = [name.strip(' \n').lower() for name in line.split(sep) if name]
        while True:
            if 'waveform' in header:
                break
            line = inp.readline()
            sep = '\t' if '\t' in line else '  '
            header = [name.strip(' \n').lower() for name in line.split(sep) if name]

        approx_col = header.index('waveform')
        approx_enum = int(inp.readline().split()[approx_col])

    approx = lalsim.GetStringFromApproximant(approx_enum)

    return approx


def consistent_max_logl(infile, max_logl=None, Neff=1000):
    """
    The max logl found is expected to asymptotically approach SNR^2/2
    in the long-chain limit, for the zero-noise realization.  With
    Guassian noise it is expected to fluctuate with a variance of
    approximately the injected SNR.

    This function checks that the max logl found is within ~x2 of the
    expected max logl based on the injected SNR.
    """
    from scipy import stats

    if not max_logl:
        max_logl = get_logl(infile).max()

    n_dim = count_dimensions(infile)

    injected_snr = get_network_snr(infile)

    # Handle normalization for marginalized likelihoods
    logl_norm = estimate_logl_normalization(infile)
    expected_max_logl = injected_snr**2/2 + logl_norm

    # Determine the quantile of the measured max_logl
    max_logl_quantile = np.power(1 - stats.distributions.gamma.cdf(expected_max_logl-logl_norm, 0.5*n_dim), Neff)

    # If 0-noise realization and no marginalizing, peform robust check of max-logl found
    if logl_norm == 0 and check_if_0noise(infile):
        if max_logl_quantile < 0.01 or max_logl_quantile > 0.99:
            print("WARNING: maximum log-likelihood = {0:.1f}".format(max_logl))
            print("  is extreme, lying at the {0:.3f} quantile".format(max_logl_quantile))
            print("  of the expected distribution for an SNR = {0:.1f} injection!".format(injected_snr))
            return False
        else:
            return True

    else:
        if injected_snr > 0.0 and abs(expected_max_logl - max_logl) > injected_snr:
            print("WARNING: maximum log-likelihood = {0:.1f}".format(max_logl))
            print("  is inconsistent with the expected ~{0:.1f}".format(0.5*injected_snr**2))
            print("  for an SNR = {0:.1f} injection!".format(injected_snr))
            return False
        else:
            return True

def extract_independent_samples(infile, max_logl=None, params=None):
    """
    Extract the independent samples from a PTMCMC output file.
    """
    n_dim = count_dimensions(infile)

    if not max_logl:
        max_logl = get_logl(infile).max()

    # Define burnin to be first time chain is within n_dim/2 of the max logl found
    delta_logl = n_dim/2.
    target_logl = max_logl - delta_logl

    with open(infile, 'rb') as inp:
        header = pass_header(inp)

        try:
            logl_col = header.index('deltalogl')
        except ValueError:
            logl_col = header.index('logl')
        logl_burnin(inp, logl_col, target_logl)

        if params is None:
            param_cols = [col for col, param in enumerate(header) if param not in non_params]
            params = [str(header[p]) for p in param_cols]
        else:
            param_cols = [header.index(param) for param in params]

        samples = np.genfromtxt(inp, usecols=param_cols, dtype=[(param, np.float) for param in params])

    N = len(samples)

    max_acl = 0
    for param in params:
        try:
            acl = autocorrelation_length_estimate(samples[param])
            max_acl =  acl if acl > max_acl else max_acl
        except ACLError:
            max_acl = 1
            break

    max_acl = np.ceil(max_acl)
    Neff = int(np.ceil(N/max_acl))

    independent_samples = np.recarray((Neff,), dtype=samples.dtype)
    for param in params:
        independent_samples[param] = samples[param][::int(max_acl)]

    return independent_samples, params


def logl_burnin(inp, logl_col, target_logl):
    """
    Given a target loglikelihood, return samples collected after
    the chain passes it for the first time.
    """
    while float(inp.readline().split()[logl_col]) < target_logl:
        continue
    return

def count_dimensions(infile, ignored_params=non_params):
    """
    Count the number of sampling dimensions.
    """
    if not ignored_params:
        ignored_params = []

    with open(infile, 'rb') as inp:
        header = pass_header(inp)

    if ignored_params:
        n_dim = len([p for p in header if p not in ignored_params])
    else:
        n_dim = len(header)

    return n_dim

def get_event_from_xml(injfile, event):
    from pylal import SimInspiralUtils

    injections = SimInspiralUtils.ReadSimInspiralFromFiles([injfile])
    injection = injections[event]

    return injection



################################################
###   Ensemble Utilities   #####################
################################################
def read_ensemble_samples(infiles, params=None, nwalkers=None, thin=1):
    # Determine number of walkers
    if nwalkers is None:
        nwalkers = 0
        for infile in infiles:
            with open(infile, 'rb') as inp:
                header = pass_header(inp)
                walker_col = header.index('walker')
                walker_ids = np.genfromtxt(inp, usecols=[walker_col], skip_footer=1)
                nwalkers += len(np.unique(walker_ids))

    if params is None:
        with open(infiles[0], 'rb') as inp:
            header = pass_header(inp)
            param_cols = [col for col, p in enumerate(header) if p not in non_params]
            params = [header[p] for p in param_cols]

    nparams = len(params)

    # Determin number of frames (shortest chain's length) on the fly
    nframes = np.inf

    acc_T = None
    pos_T = None
    for infile in infiles:
        with open(infile, 'rb') as inp:
            header = pass_header(inp)
            cols = [header.index('walker')]
            [cols.append(header.index(p)) for p in params]
            pos = np.genfromtxt(inp, usecols=cols, skip_footer=1)

            if pos_T is None:
                nframes = np.count_nonzero(pos[:, 0] == pos[0, 0])
                pos_T = np.zeros((nframes, nwalkers, nparams))

            for walker in np.unique(pos[:, 0]):
                nframes = min(nframes, np.count_nonzero(pos[:, 0] == walker))
                pos_T[:nframes, walker] = pos[pos[:, 0] == walker][:nframes, 1:]

    pos_T = pos_T[:nframes:thin, :, :]

    return pos_T, params


################################################
###   Animated triangle!   #####################
################################################
def animate_triangle(pos_T, labels=None, truths=None, samps_per_frame=10, fps=30, rough_length=10.0, outname='triangle.mp4'):
    from matplotlib import animation
    import triangle

    nframes, nwalkers, ndim = pos_T.shape

    final_bins = 50  #number of bins covering final posterior
    # Use last time step to get y-limits of histograms
    bins = []
    ymaxs = []
    for x in range(ndim):
        dx = (pos_T[-1,:,x].max() - pos_T[-1,:,x].min())/final_bins
        nbins = int((pos_T[0,:,x].max() - pos_T[0,:,x].min())/dx)
        bins.append(np.linspace(pos_T[0,:,x].min(), pos_T[0,:,x].max(), nbins+1)[:-1])
        hist, _ = np.histogram(pos_T[-1,:,x], bins=bins[-1], normed=True)
        ymaxs.append(1.1*max(hist))

    # Use the first time sample as the initial frame
    fig = triangle.corner(pos_T[0], labels=labels, plot_contours=False, truths=truths)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for x in range(ndim):
        axes[x,x].set_ylim(top=ymaxs[x])

    # Determine number of frames
    thin_factor = int(nframes/rough_length)/fps
    if thin_factor > 1:
        pos_T = pos_T[::thin_factor]
        samps_per_frame *= thin_factor
    samps_per_sec = fps * samps_per_frame

    # Make the movie
    anim = animation.FuncAnimation(fig, update_triangle, frames=xrange(len(pos_T)), blit=True,
                                             fargs=(pos_T, fig, bins, truths))
    return anim

def update_triangle(i, data, fig, bins, truths=None):
    ndim = data.shape[-1]
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Update histograms along diagonal
    for x in range(ndim):
        ax = axes[x, x]

        # Save bins and y-limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Clean current histrogram while keeping ticks
        while len(ax.patches) > 0:
            ax.patches[0].remove()

        ax.hist(data[i,:,x], range=xlim, bins=bins[x], histtype='step', normed=True, color='k')
        ax.set_ylim(*ylim)
        if truths is not None:
                ax.axvline(truths[x], color="#4682b4")

    # Update scatter plots
    for x in range(1, ndim):
        for y in range(x):
            ax = axes[x, y]
            line = ax.get_lines()[0]
            line.set_data(data[i,:,y], data[i,:,x])

    return fig,


def update_scatter(i, data, scat, acc=None, proj=None):
    scat.set_label("{}".format(i))
    if proj == '3d':
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import juggle_axes

        xs = data[i,:,0]
        ys = data[i,:,1]
        zs = data[i,:,2]
        scat._offsets3d = juggle_axes(xs, ys, zs, 'z')

    else:
        scat.set_offsets(data[i,:,:])

    if acc is not None and proj is None:
        scat.set_color(cm.RdYlGn(acc[i,:,0]))

    #plt.legend()
    return scat,


def thin_ensemble(pos_T, params):
    nsteps, nwalkers, ndim = pos_T.shape
    #acc_rate = np.mean(pos_T[-nsteps/10:, :, params.index("acceptance_rate")], axis=1)
    acc_rate = 0.1
    return pos_T[-nsteps/2:, ...]


################################################
###   Plotting Utilities   #####################
################################################

def make_triangle(sample_array, params, injdict=None):
    """
    Make a triangle plot of the sampling parameters.
    """
    # Use Agg backend to avoid needing an X-server
    import matplotlib
    matplotlib.use('Agg')

    import triangle

    labels = [plot_label(param) for param in params]

    if injdict:
        true_vals = [injdict[param] for param in params]
    else:
        true_vals = None

    fig = triangle.corner(sample_array, labels=labels, truths = true_vals)

    return fig

def plot_skymap(output, skypost, pixresol=np.pi/180.0, nest=True):
    from matplotlib import pyplot as plt
    import healpy as hp
    try:
        from lalinference.cmap import cylon as cmap
    except ImportError:
        cmap = plt.cmap('gray')

    cmap.set_under(color='w')

    nside = 1
    while hp.nside2resol(nside) > pixresol:
        nside *= 2

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside), dtype=np.int), nest=nest)
    pixels = np.column_stack((phis, np.pi/2.0 - thetas))

    pix_post = skypost.posterior(pixels)

    plt.clf()
    hp.mollview(pix_post, cmap=cmap, title="", nest=nest)
    plt.savefig(output)

def make_skymap(pts, maxpts=None, trials=50, outname='skymap.png'):
    from sky_area import sky_area_clustering as sac
    if maxpts is not None:
        pts = np.random.permutation(pts)[:maxpts, :]

    for i in range(50):
        try:
            skypost = sac.ClusteredSkyKDEPosterior(pts)
            break
        except:
            skypost = None
            continue
    plot_skymap(outname, skypost)


def add_logl_plot(fig, logls, SNR=0.0, dim=None, nskip=100, logl_norm=0.0, burned_in=None):
    """
    Add a plot of the log-likelihood series from each chain
    to the triangle plot contained in "fig".
    """
    # Put logl plot in upper-right, which should be empty
    bg_color = None
    if burned_in is not None:
        bg_color = '#00FF00' if burned_in else '#FF0000'

    ax = fig.add_subplot(322, axisbg=bg_color)
    ax.patch.set_alpha(0.05)

    ax.set_xlabel(plot_label('samples after burnin'))
    ax.set_ylabel(plot_label('logl'))

    max_logl = 0
    for logl in logls:
        ax.plot(nskip*np.arange(len(logl)), logl, linewidth=0.0, marker=',')
        max_logl = logl.max() if logl.max() > max_logl else max_logl

    # Add guides for the expected with of the logl samples
    ax.axhline(max_logl, *ax.get_xlim(), ls='-')

    expected_max_logl = None
    if SNR > 0.0:
        expected_max_logl = logl_norm + 0.5*SNR**2
        ax.axhline(expected_max_logl, *ax.get_xlim(), ls='-', c='c')

        if dim is not None:
            ax.axhline(expected_max_logl - 2*dim, *ax.get_xlim(), ls='--', c='c')
            ax.axhline(max_logl - 2*dim, *ax.get_xlim(), ls='--')
            ax.set_ylim(ymin=max_logl - 4*dim)

    ax.set_ylim(ymax=1.05*max(max_logl, expected_max_logl))

def plot_label(param):
    """
    A lookup table for plot labels.
    """
    m1_names = ['mass1', 'm1']
    m2_names = ['mass2', 'm2']
    mc_names = ['mc','mchirp','chirpmass']
    eta_names = ['eta','massratio','sym_massratio']
    q_names = ['q','asym_massratio']
    iota_names = ['iota','incl','inclination']
    dist_names = ['dist','distance']
    ra_names = ['rightascension','ra']
    dec_names = ['declination','dec']
    phase_names = ['phi_orb', 'phi', 'phase']
  
    labels={
        'm1':r'$m_1\,(\mathrm{M}_\odot)$',
        'm2':r'$m_2\,(\mathrm{M}_\odot)$',
        'mc':r'$\mathcal{M}\,(\mathrm{M}_\odot)$',
        'eta':r'$\eta$',
        'q':r'$q$',
        'mtotal':r'$M_\mathrm{total}\,(\mathrm{M}_\odot)$',
        'spin1':r'$S_1$',
        'spin2':r'$S_2$',
        'a1':r'$a_1$',
        'a2':r'$a_2$',
        'theta1':r'$\theta_1\,(\mathrm{rad})$',
        'theta2':r'$\theta_2\,(\mathrm{rad})$',
        'theta_spin1':r'$\theta_1\,(\mathrm{rad})$',
        'theta_spin2':r'$\theta_2\,(\mathrm{rad})$',
        'phi1':r'$\phi_1\,(\mathrm{rad})$',
        'phi2':r'$\phi_2\,(\mathrm{rad})$',
        'phi_spin1':r'$\phi_1\,(\mathrm{rad})$',
        'phi_spin2':r'$\phi_2\,(\mathrm{rad})$',
        'chi':r'$\chi$',
        'tilt1':r'$t_1\,(\mathrm{rad})$',
        'tilt2':r'$t_2\,(\mathrm{rad})$',
        'costilt1':r'$\mathrm{cos}(t_1)$',
        'costilt2':r'$\mathrm{cos}(t_2)$',
        'iota':r'$\iota\,(\mathrm{rad})$',
        'cosiota':r'$\mathrm{cos}(\iota)$',
        'time':r'$t_\mathrm{c}\,(\mathrm{s})$',
        'dist':r'$d_\mathrm{L}\,(\mathrm{Mpc})$',
        'logdistance':r'$\mathrm{log}(d_\mathrm{L}\,(\mathrm{Mpc}))$',
        'ra':r'$\alpha (\mathrm{rad})$',
        'dec':r'$\delta (\mathrm{rad})$',
        'sindec':r'$\mathrm{sin}(\delta)$',
        'phase':r'$\phi\,(\mathrm{rad})$',
        'psi':r'$\psi\,(\mathrm{rad})$',
        'theta_jn':r'$\theta_\mathrm{JN}\,(\mathrm{rad})$',
        'costheta_jn':r'$\mathrm{cos}(\theta_\mathrm{JN})$',
        'beta':r'$\beta\,(\mathrm{rad})$',
        'cosbeta':r'$\mathrm{cos}(\beta)$',
        'phi_jl':r'$\phi_\mathrm{JL}\,(\mathrm{rad})$',
        'phi12':r'$\phi_\mathrm{12}\,(\mathrm{rad})$',
        'logl':r'$\mathrm{log}(\mathcal{L})$',
        'deltalogl':r'$\Delta\mathrm{log}(\mathcal{L})$',
        'h1_end_time':r'$t_\mathrm{H}$',
        'l1_end_time':r'$t_\mathrm{L}$',
        'v1_end_time':r'$t_\mathrm{V}$',
        'h1l1_delay':r'$\Delta t_\mathrm{HL}$',
        'h1v1_delay':r'$\Delta t_\mathrm{HV}$',
        'l1v1_delay':r'$\Delta t_\mathrm{LV}$',
        'lambdat' : r'$\tilde{\Lambda}$',
        'dlambdat': r'$\delta \tilde{\Lambda}$',
        'lambda1' : r'$\lambda_1$',
        'lambda2': r'$\lambda_2$',
        'lam_tilde' : r'$\tilde{\Lambda}$',
        'dlam_tilde': r'$\delta \tilde{\Lambda}$'}
  
    # Handle cases where multiple names have been used
    if param in m1_names:
        param = 'm1'
    elif param in m2_names:
        param = 'm2'
    elif param in mc_names:
        param = 'mc'
    elif param in eta_names:
        param = 'eta'
    elif param in q_names:
        param = 'q'
    elif param in iota_names:
        param = 'iota'
    elif param in dist_names:
        param = 'dist'
    elif param in ra_names:
        param = 'ra'
    elif param in dec_names:
        param = 'dec'
    elif param in phase_names:
        param = 'phase'
  
    try:
        label = labels[param]
    except KeyError:
        # Use simple string if no formated label is available for param
        label = param
  
    return label


################################################
###   Autocorrelation Calculation   ############
################################################

class ACLError(Exception):
    def __init__(self, *args):
        super(ACLError, self).__init__(*args)


def autocorrelation(series):
    """Returns an estimate of the autocorrelation function of a given
    series.  Returns only the positive-lag portion of the ACF,
    normalized so that the zero-th element is 1."""
    from scipy import signal

    x=series-np.mean(series)
    y=np.conj(x[::-1])

    acf=np.fft.ifftshift(signal.fftconvolve(y,x,mode='full'))

    N=series.shape[0]

    acf = acf[0:N]

    return acf/acf[0]


def autocorrelation_length_estimate(series, acf=None, M=5, K=2):
    """Attempts to find a self-consistent estimate of the
    autocorrelation length of a given series.

    If C(tau) is the autocorrelation function (normalized so C(0) = 1,
    for example from the autocorrelation procedure in this module),
    then the autocorrelation length is the smallest s such that

    1 + 2*C(1) + 2*C(2) + ... + 2*C(M*s) < s

    In words: the autocorrelation length is the shortest length so
    that the sum of the autocorrelation function is smaller than that
    length over a window of M times that length.

    The maximum window length is restricted to be len(series)/K as a
    safety precaution against relying on data near the extreme of the
    lags in the ACF, where there is a lot of noise.  Note that this
    implies that the series must be at least M*K*s samples long in
    order to get a reliable estimate of the ACL.

    If no such s can be found, raises ACLError; in this case it is
    likely that the series is too short relative to its true
    autocorrelation length to obtain a consistent ACL estimate."""

    if acf is None:
      acf=autocorrelation(series)
    acf[1:] *= 2.0

    imax=int(acf.shape[0]/K)

    # Cumulative sum and ACL length associated with each window
    cacf=np.cumsum(acf)
    s=np.arange(1, cacf.shape[0]+1)/float(M)

    # Find all places where cumulative sum over window is smaller than
    # associated ACL.
    estimates=np.flatnonzero(cacf[:imax] < s[:imax])

    if estimates.shape[0] > 0:
        # Return the first index where cumulative sum is smaller than
        # ACL associated with that index's window
        return s[estimates[0]]
    else:
        # Cannot find self-consistent ACL estimate.
        raise ACLError('autocorrelation length too short for consistent estimate')


################################################
###   Injection Value Extraction   #############
################################################

def extract_inj_vals(sim_inspiral_event, f_ref=100):
    a1, a2, spin1z, spin2z, theta_jn, phi_jl, tilt1, tilt2, phi12, beta = calculate_injected_sys_frame_params(sim_inspiral_event, f_ref)
    injvals={
        'mc'          : sim_inspiral_event.mchirp,
        'q'           : sim_inspiral_event.mass2/sim_inspiral_event.mass1,
        'm1'          : sim_inspiral_event.mass1,
        'm2'          : sim_inspiral_event.mass2,
        'time'        : float(sim_inspiral_event.get_end()),
        'phi_orb'     : sim_inspiral_event.coa_phase,
        'dist'        : sim_inspiral_event.distance,
        'logdistance' : np.log(sim_inspiral_event.distance),
        'ra'          : sim_inspiral_event.longitude,
        'dec'         : sim_inspiral_event.latitude,
        'sindec'      : np.sin(sim_inspiral_event.latitude),
        'psi'         : np.mod(sim_inspiral_event.polarization, np.pi),
        'a1'          : a1,
        'a2'          : a2,
        'spin1'       : spin1z,
        'spin2'       : spin2z,
        'phi12'       : phi12,
        'tilt1'       : tilt1,
        'tilt2'       : tilt2,
        'costilt1'    : np.cos(tilt1),
        'costilt2'    : np.cos(tilt2),
        'theta_jn'    : theta_jn,
        'costheta_jn' : np.cos(theta_jn),
        'beta'        : beta,
        'cosbeta'     : np.cos(beta),
        'phi12'       : phi12,
        'phi_jl'      : phi_jl}
    return injvals


def calculate_injected_sys_frame_params(sim_inspiral_event, f_ref=100.0):
    L  = orbital_momentum(f_ref, sim_inspiral_event.mchirp, sim_inspiral_event.inclination)
    S1 = np.hstack((sim_inspiral_event.spin1x, sim_inspiral_event.spin1y, sim_inspiral_event.spin1z))
    S2 = np.hstack((sim_inspiral_event.spin2x, sim_inspiral_event.spin2y, sim_inspiral_event.spin2z))

    a1 = np.sqrt(np.sum(S1 * S1))
    a2 = np.sqrt(np.sum(S2 * S2))

    S1 *= sim_inspiral_event.mass1*sim_inspiral_event.mass1
    S2 *= sim_inspiral_event.mass2*sim_inspiral_event.mass2

    J = L + S1 + S2

    tilt1 = array_ang_sep(L, S1) if not all([i==0.0 for i in S1]) else 0.0
    tilt2 = array_ang_sep(L, S2) if not all([i==0.0 for i in S2]) else 0.0

    if sim_inspiral_event.spin1x == 0.0 and sim_inspiral_event.spin1y == 0.0:
        spin1z = sim_inspiral_event.spin1z
    else:
        spin1z = a1 * np.cos(tilt1)

    if sim_inspiral_event.spin2x == 0.0 and sim_inspiral_event.spin2y == 0.0:
        spin2z = sim_inspiral_event.spin2z
    else:
        spin2z = a2 * np.cos(tilt2)

    theta_jn = array_polar_ang(J)

    # Need to do rotations of XLALSimInspiralTransformPrecessingInitialConditioin inverse order to go in the L frame
    # first rotation: bring J in the N-x plane, with negative x component
    phi0 = np.arctan2(J[1], J[0])
    phi0 = np.pi - phi0

    J = ROTATEZ(phi0, J[0], J[1], J[2])
    L = ROTATEZ(phi0, L[0], L[1], L[2])
    S1 = ROTATEZ(phi0, S1[0], S1[1], S1[2])
    S2 = ROTATEZ(phi0, S2[0], S2[1], S2[2])

    # now J in in the N-x plane and form an angle theta_jn with N, rotate by -theta_jn around y to have J along z
    J = ROTATEY(theta_jn,J[0],J[1],J[2])
    L = ROTATEY(theta_jn,L[0],L[1],L[2])
    S1 = ROTATEY(theta_jn,S1[0],S1[1],S1[2])
    S2 = ROTATEY(theta_jn,S2[0],S2[1],S2[2])

    # J should now be || z and L should have a azimuthal angle phi_jl
    phi_jl = np.arctan2(L[1], L[0])
    if phi_jl<0.:
        phi_jl+=2.0*np.pi

    # bring L in the Z-X plane, with negative x
    J = ROTATEZ(phi_jl, J[0], J[1], J[2])
    L = ROTATEZ(phi_jl, L[0], L[1], L[2])
    S1 = ROTATEZ(phi_jl, S1[0], S1[1], S1[2])
    S2 = ROTATEZ(phi_jl, S2[0], S2[1], S2[2])

    theta0 = array_polar_ang(L)
    J = ROTATEY(theta0, J[0], J[1], J[2])
    L = ROTATEY(theta0, L[0], L[1], L[2])
    S1 = ROTATEY(theta0, S1[0], S1[1], S1[2])
    S2 = ROTATEY(theta0, S2[0], S2[1], S2[2])

    # The last rotation is useless as it won't change the differenze in spins' azimuthal angles
    phi1 = np.arctan2(S1[1],S1[0])
    phi2 = np.arctan2(S2[1],S2[0])
    if phi2 < phi1:
        phi12 = phi2 - phi1 + 2.*np.pi
    else:
        phi12 = phi2 - phi1

    beta  = array_ang_sep(J, L)

    return a1, a2, spin1z, spin2z, theta_jn, phi_jl, tilt1, tilt2, phi12, beta

def ROTATEZ(angle, vx, vy, vz):
    # This is the ROTATEZ in LALSimInspiral.c.
    tmp1 = vx*np.cos(angle) - vy*np.sin(angle);
    tmp2 = vx*np.sin(angle) + vy*np.cos(angle);
    return np.asarray([tmp1,tmp2,vz])

def ROTATEY(angle, vx, vy, vz):
    # This is the ROTATEY in LALSimInspiral.c
    tmp1 = vx*np.cos(angle) + vz*np.sin(angle);
    tmp2 = - vx*np.sin(angle) + vz*np.cos(angle);
    return np.asarray([tmp1,vy,tmp2])

def array_ang_sep(vec1, vec2):
    """
    Find angles between vectors in rows of numpy arrays.
    """
    vec1_mag = np.sqrt(array_dot(vec1, vec1))
    vec2_mag = np.sqrt(array_dot(vec2, vec2))
    return np.arccos(array_dot(vec1, vec2)/(vec1_mag*vec2_mag))

def sph2cart(r,theta,phi):
    """
    Utiltiy function to convert r,theta,phi to cartesian co-ordinates.
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def array_dot(vec1, vec2):
    """
    Calculate dot products between vectors in rows of numpy arrays.
    """
    if vec1.ndim==1:
        product = (vec1*vec2).sum()
    else:
        product = (vec1*vec2).sum(axis=1).reshape(-1,1)
    return product

def array_polar_ang(vec):
    """
    Find polar angles of vectors in rows of a numpy array.
    """
    if vec.ndim==1:
        z = vec[2]
    else:
        z = vec[:,2].reshape(-1,1)
    norm = np.sqrt(array_dot(vec,vec))
    return np.arccos(z/norm)


def orbital_momentum(fref, mc, inclination):
    """
    Calculate orbital angular momentum vector.
    Note: The units of Lmag are different than what used in lalsimulation.
    Mc must be called in units of Msun here.

    Note that if one wants to build J=L+S1+S2 with L returned by this function, S1 and S2
    must not get the Msun^2 factor.
    """
    from lal import MTSUN_SI
    Lmag = np.power(mc, 5.0/3.0) / np.power(np.pi * MTSUN_SI * fref, 1.0/3.0)
    Lx, Ly, Lz = sph2cart(Lmag, inclination, 0.0)
    return np.hstack((Lx,Ly,Lz))



################################################
###  Main Routine    ###########################
################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Light-weight post process one or many PTMCMC output files.")
    parser.add_argument('--inj', metavar='inj.xml', type=str, help='Injection XML file.')
    parser.add_argument('--event', metavar='eventnum', type=str, help='Number of injection in XML file.')
    parser.add_argument('--inj-fref', metavar='eventnum', type=float, default=100.0, help='Reference frequency of injection.')
    parser.add_argument('--user', metavar='user', type=str, help='Specify the user who conducted the analysis.')
    parser.add_argument('--outname', metavar='outname', type=str, help='Specify an output name and file type (e.g. posterior.pdf).')
    parser.add_argument('--param', action="append", help='Parameter(s) to plot.')
    parser.add_argument('samples', metavar='PTMCMC.output.*.00', type=str, nargs='+',
                               help='PTMCMC output file(s).')

    args = parser.parse_args()

    sample_array = None
    logls = []
    burned_in = True
    infile = args.samples[0]

    try:
        sample_array, params = read_ensemble_samples(args.samples, params=args.param)
        #sample_array = np.vstack(thin_ensemble(sample_array, params))
        sample_array = np.vstack(sample_array)

    except (RuntimeError, ValueError):
        for infile in args.samples:
            logl = get_logl(infile)
            max_logl = logl.max()
            logls.append(logl)

            #try:
            samples, params = extract_independent_samples(infile, params=args.param)
            #except (TypeError, IndexError):
            #    print "continue"
            #    continue

            print("{} independent samples collected from {}.".format(len(samples), infile))

            # Check that max logl is consistent with the injected network SNR
            burned_in = consistent_max_logl(infile, max_logl, Neff=len(samples))

            if params is None:
                params = samples.dtype.names

            if sample_array is None:
                sample_array = samples.view(float).reshape(-1, len(params))
            else:
                sample_array = np.concatenate([sample_array, samples.view(float).reshape(-1, len(params))])

    norm = estimate_logl_normalization(infile)

    if args.inj:
        event = extract_event_number(infile) if args.event is None else int(args.event)
        inj = get_event_from_xml(args.inj, event)
        injvals = extract_inj_vals(inj, args.inj_fref)
    else:
        event = None
        injvals = None

    if args.outname is not None:
        outname = args.outname
    else:
        try:
            approx = get_approx(infile)
            outname = generate_default_output_filename(approx=approx, event=event)
        except (ImportError, ValueError):
            outname = "posterior.png"

    fig = make_triangle(sample_array, params, injdict = injvals)
    add_logl_plot(fig, logls, SNR=get_network_snr(infile), dim=len(params), logl_norm=norm, burned_in=burned_in)
    fig.savefig(outname)

    skypts = np.column_stack([sample_array[:, params.index('ra')],
                              sample_array[:, params.index('dec')]])
    make_skymap(skypts)

    with open('posterior_samples.dat', 'w') as outp:
        outp.write(' '.join(params)+'\n')
        np.savetxt(outp, sample_array)
