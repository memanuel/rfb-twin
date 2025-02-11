# Make symlinks for rincflo
ln -sf /home/michael/Harvard/rfb/rincflo/build2d/rfb2d.gnu.MPI.EB.ex /usr/local/bin/rfb2d-mpi
ln -sf /home/michael/Harvard/rfb/rincflo/build2d/rfb2d.gnu.DEBUG.MPI.EB.ex /usr/local/bin/rfb2d-debug-mpi

# Make symlinks for rfbsim and related scripts
ln -sf /home/michael/Harvard/rfb/rfbsh/rfb_clean_sim.sh /usr/local/bin/rfb_clean_sim
ln -sf /home/michael/Harvard/rfb/rfbsh/rfb_clean_stage.sh /usr/local/bin/rfb_clean_stage

# Make symlinks for amrvis
ln -sf /home/michael/SoftwareBuild/Amrvis/amrvis1d.gnu.ex /usr/local/bin/amrvis1d
ln -sf /home/michael/SoftwareBuild/Amrvis/amrvis2d.gnu.ex /usr/local/bin/amrvis2d
ln -sf /home/michael/SoftwareBuild/Amrvis/amrvis3d.gnu.ex /usr/local/bin/amrvis3d

# Make symlink for rfbvis (a.k.a. opengl, to visualize geometry files)
ln -sf /home/michael/Harvard/rfb/opengl_configs/opengl /usr/local/bin/rfbvis
