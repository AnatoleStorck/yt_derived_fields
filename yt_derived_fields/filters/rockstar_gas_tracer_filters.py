# Generating particle filters to use gas tracers with the YT api of Rockstar

# Author: Anatole Storck


import yt
import unyt as u
from yt import mylog

import glob
import numpy as np


def get_Npart(ds):
    # Quickly look up the number of DM and tracer particles in the dataset.
    directory = ds.directory
    header = glob.glob(directory + "/header_*")[0]

    ptypes = np.loadtxt(open(header).readlines()[1:-2], dtype=str, usecols=0)
    Npart = np.loadtxt(open(header).readlines()[1:-2], dtype=int, usecols=1)

    for N, ptype in zip(Npart, ptypes, strict=False):
        if ptype == "gas_tracer":
            N_gas_tracer = N
        if ptype == "DM":
            N_dm = N

    return N_gas_tracer, N_dm


@yt.particle_filter(requires=["particle_family", "particle_mass"], filtered_type="gas_tracer")
def gas_tracer_noNaN(field, data):
    """
    Filter to select gas tracers that have a valid cell index.
    This error occurs when the chunk the particle is in does not include the actual cell
    the particle is in, which sometimes happens on the boundaries of the chunk domain.
    """
    gas_tracer_indexes = data["gas_tracer", "cell_index"]
    isnan_cell_index = np.isnan(gas_tracer_indexes)

    # Return a boolean array where True indicates valid gas tracers
    return ~isnan_cell_index


def particle_position_redefine_for_gas_tracers(field, data):
    """
    Redefine "particle_position" for the subset of io particles which
    are gas tracers. If there is more than one tracer in a cell, we reposition
    the tracers to space them out evenly in the cell.
    """
    io_pos = data["io", "particle_position"]
    io_fam = data["io", "particle_family"]

    gas_tracer_filter = io_fam == 0
    gas_tracer_indexes = data["gas_tracer", "cell_index"]
    isnan_cell_index = np.isnan(gas_tracer_indexes)
    gas_tracer_filter[gas_tracer_filter] &= ~isnan_cell_index

    if gas_tracer_filter.sum() > 0:
        tracer_cell_index = data["gas_tracer_noNaN", "cell_index"]
        tracer_cell_dx = np.cbrt(data["gas_tracer_noNaN", "cell_gas_volume"]).to("Mpccm/h")

        tracer_pos = io_pos[gas_tracer_filter]

        new_tracer_pos = tracer_pos.copy()

        if len(tracer_pos) > 1:
            for cell_index in np.unique(tracer_cell_index):
                cell_mask = tracer_cell_index == cell_index
                n_particles_in_cell = np.sum(cell_mask)
                if n_particles_in_cell > 1:
                    # center of cell is given by one of the tracer positions
                    # (all are at the center), so just need one
                    cell_center = tracer_pos[cell_mask][0, :]
                    cell_dx = tracer_cell_dx[cell_mask][0]

                    # -------- Generate new positions for the tracers in this cell --------#
                    # Find the number of particles along each axis (as close to a cube as possible)
                    n_per_axis = int(np.ceil(n_particles_in_cell ** (1 / 3)))

                    inner_boundary = cell_dx / 3 * (1 - 1 / n_per_axis)

                    # Generate grid points
                    grid = np.linspace(-inner_boundary, inner_boundary, n_per_axis)
                    x, y, z = np.meshgrid(grid, grid, grid)
                    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

                    # Assign random particle positions from the grid points
                    # as not to bias the particle distribution towards any axis
                    np.random.shuffle(grid_points)
                    new_pos = grid_points[:n_particles_in_cell] + cell_center
                    # ---------------------------------------------------------------------#

                    new_tracer_pos[cell_mask] = new_pos

        io_pos[gas_tracer_filter] = new_tracer_pos

    return io_pos


def particle_position_redefine_for_gas_tracers_x(field, data):
    io_pos = data["io", "particle_position_redefine_for_gas_tracers"]
    return io_pos[:, 0]  # Return only the x-component of the position


def particle_position_redefine_for_gas_tracers_y(field, data):
    io_pos = data["io", "particle_position_redefine_for_gas_tracers"]
    return io_pos[:, 1]  # Return only the y-component of the position


def particle_position_redefine_for_gas_tracers_z(field, data):
    io_pos = data["io", "particle_position_redefine_for_gas_tracers"]
    return io_pos[:, 2]  # Return only the z-component of the position


def particle_velocity_redefine_for_gas_tracers(field, data):
    """
    Redefine "particle_velocity" for the subset of io particles which
    are gas tracers. This function deposits the gas velocity of the cell
    where the gas tracer is located onto the gas tracer particle
    (gas tracers do not have their own velocity). The velocity is perturbed
    by a small amount to avoid the detection of duplicate particles by rockstar.
    """
    io_vel = data["io", "particle_velocity"]
    io_fam = data["io", "particle_family"]

    gas_tracer_filter = io_fam == 0
    gas_tracer_indexes = data["gas_tracer", "cell_index"]
    isnan_cell_index = np.isnan(gas_tracer_indexes)
    gas_tracer_filter[gas_tracer_filter] &= ~isnan_cell_index

    if gas_tracer_filter.sum() > 0:
        tracer_cell_vel_x = data["gas_tracer_noNaN", "cell_gas_velocity_x"].to("km/s")
        tracer_cell_vel_y = data["gas_tracer_noNaN", "cell_gas_velocity_y"].to("km/s")
        tracer_cell_vel_z = data["gas_tracer_noNaN", "cell_gas_velocity_z"].to("km/s")

        new_tracer_vel = np.column_stack([tracer_cell_vel_x, tracer_cell_vel_y, tracer_cell_vel_z])

        perturbation = np.random.uniform(-1, 1, size=new_tracer_vel.shape) * 1e-3 * u.km / u.s

        new_tracer_vel[:, 0] += perturbation[:, 0]
        new_tracer_vel[:, 1] += perturbation[:, 1]
        new_tracer_vel[:, 2] += perturbation[:, 2]

        io_vel[gas_tracer_filter] = new_tracer_vel

    return io_vel


def particle_velocity_redefine_for_gas_tracers_x(field, data):
    io_vel = data["io", "particle_velocity_redefine_for_gas_tracers"]
    return io_vel[:, 0]  # Return only the x-component of the velocity


def particle_velocity_redefine_for_gas_tracers_y(field, data):
    io_vel = data["io", "particle_velocity_redefine_for_gas_tracers"]
    return io_vel[:, 1]  # Return only the y-component of the velocity


def particle_velocity_redefine_for_gas_tracers_z(field, data):
    io_vel = data["io", "particle_velocity_redefine_for_gas_tracers"]
    return io_vel[:, 2]  # Return only the z-component of the velocity


def particle_identity_rescaled(field, data):
    """
    Since the particle IDs for each particle type are not unique across
    different particle families, we need to rescale the IDs so that rockstar
    can uniquely identify each particle. We do this by rescaling the IDs
    based on the number of particles in DM and gas tracers, as those numbers
    are set at the start of the simulation, so the rescaling is the same across
    outputs.
    """

    N_gas_tracer, N_dm = get_Npart(data.ds)

    io_fam = data["io", "particle_family"]
    io_id = data["io", "particle_identity"]

    gas_tracer_filter = io_fam == 0
    star_filter = io_fam == 2

    # We multiply by 8 here as the IDs can go bigger than the number of particles
    io_id[gas_tracer_filter] = io_id[gas_tracer_filter] + (N_dm * 8)
    io_id[star_filter] = io_id[star_filter] + (N_dm * 8 * 8) + N_gas_tracer

    return io_id


@yt.particle_filter(requires=["particle_family", "particle_mass"], filtered_type="io")
def dm_star_gastracer_union(pfilter, data):
    dm_filter = data[pfilter.filtered_type, "particle_family"] == 1

    mask = dm_filter

    # # Need to check there are DM particles to take the minimum mass from
    # p_mass = data[pfilter.filtered_type, 'particle_mass']
    # if np.sum(dm_filter) > 0:
    #     #dm_Mmin = p_mass[dm_filter].to("Msun").min().value
    #     high_res_dm_filter = np.isclose(p_mass.to("Msun").value, ref_DM_mass.to("Msun").value, atol=0.0, rtol=1.25)

    #     filter &= high_res_dm_filter

    star_filter = data[pfilter.filtered_type, "particle_family"] == 2
    mask |= star_filter

    gas_tracer_filter = data[pfilter.filtered_type, "particle_family"] == 0
    gas_tracer_indexes = data["gas_tracer", "cell_index"]
    isnan_cell_index = np.isnan(gas_tracer_indexes)
    gas_tracer_filter[gas_tracer_filter] &= ~isnan_cell_index

    mask |= gas_tracer_filter

    return mask


@yt.particle_filter(requires=["particle_family", "particle_mass"], filtered_type="io")
def dm_star_union(pfilter, data):
    dm_filter = data[pfilter.filtered_type, "particle_family"] == 1
    mask = dm_filter

    star_filter = data[pfilter.filtered_type, "particle_family"] == 2
    mask |= star_filter

    return mask


def setup_dm_gas_tracers_field(ds, use_stars=True, use_gas_tracers=True):
    if use_gas_tracers:
        # start by initialzing grabbing cell data for the tracers
        ds.add_mesh_sampling_particle_field(("gas", "dx"), ptype="gas_tracer")

        # generate new particle type filtering out NaN cell indices
        ds.add_particle_filter("gas_tracer_noNaN")

        # grab the gas tracer cell data
        ds.add_mesh_sampling_particle_field(("gas", "volume"), ptype="gas_tracer_noNaN")
        for axis in "xyz":
            ds.add_mesh_sampling_particle_field(("gas", f"velocity_{axis}"), ptype="gas_tracer_noNaN")

        # Redefine particle positions for gas tracers
        ds.add_field(
            ("io", "particle_position_redefine_for_gas_tracers"),
            sampling_type="particle",
            function=particle_position_redefine_for_gas_tracers,
            units="Mpccm/h",
        )
        for axis, f in zip(
            "xyz",
            (
                particle_position_redefine_for_gas_tracers_x,
                particle_position_redefine_for_gas_tracers_y,
                particle_position_redefine_for_gas_tracers_z,
            ),
            strict=False,
        ):
            ds.add_field(
                ("io", f"particle_position_redefine_for_gas_tracers_{axis}"),
                sampling_type="particle",
                function=f,
                units="Mpccm/h",
            )

        # Redefine particle velocities for gas tracers
        ds.add_field(
            ("io", "particle_velocity_redefine_for_gas_tracers"),
            sampling_type="particle",
            function=particle_velocity_redefine_for_gas_tracers,
            units="km/s",
        )
        for axis, f in zip(
            "xyz",
            (
                particle_velocity_redefine_for_gas_tracers_x,
                particle_velocity_redefine_for_gas_tracers_y,
                particle_velocity_redefine_for_gas_tracers_z,
            ),
            strict=False,
        ):
            ds.add_field(
                ("io", f"particle_velocity_redefine_for_gas_tracers_{axis}"),
                sampling_type="particle",
                function=f,
                units="km/s",
            )

    if use_stars or use_gas_tracers:
        # Rescale particle identities for DM and stars by N_dm and N_gas_tracer
        ds.add_field(
            ("io", "particle_identity_rescaled"),
            sampling_type="particle",
            function=particle_identity_rescaled,
            units="dimensionless",
        )

    # Add the union of DM, star, and gas tracer particles (taken from io particles)
    if use_stars:
        mylog.info("Adding particle filter for DM and stars.")
        ds.add_particle_filter("dm_star_union")
    if use_gas_tracers:
        mylog.info("Adding particle filter for DM, stars, and gas tracers.")
        ds.add_particle_filter("dm_star_gastracer_union")
