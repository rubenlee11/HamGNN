/*****************************************************************************

  Ver. 3.9 (26/July/2019)

  OpenMX (Open source package for Material eXplorer) is a program package
  for linear scaling density functional calculations of large-scale materials.
  Almost time-consuming parts can be performed in O(N) operations where N is
  the number of atoms (or basis orbitals). Thus, the program might be useful
  for studies of large-scale materials.
  The distribution of this program package follows the practice of
  the GNU General Public Licence (GPL).

  OpenMX is based on

   *  Local density and generalized gradient approximation (LDA, LSDA, GGA)
      to the exchange-corellation term
   *  Norm-conserving pseudo potentials
   *  Variationally optimized pseudo atomic basis orbitals
   *  Solution of Poisson's equation using FFT
   *  Evaluation of two-center integrals using Fourier transformation
   *  Evaluation of three-center integrals using fixed real space grids
   *  Simple mixing, direct inversion in the interative subspace (DIIS),
      and Guaranteed-reduction Pulay's methods for SCF calculations.
   *  Solution of the eigenvalue problem using O(N) methods
   *  ...

  See also our website (http://www.openmx-square.org/)
  for recent developments.


    **************************************************************
     Copyright

     Taisuke Ozaki

     Present (23/Sep./2019) official address

       Institute for Solid State Physics, University of Tokyo,
       Kashiwanoha 5-1-5, Kashiwa, Chiba 277-8581, Japan

       e-mail: t-ozaki@issp.u-tokyo.ac.jp
    **************************************************************

*****************************************************************************/

/**********************************************************************
  openmx.c:

     openmx.c is the main routine of OpenMX.

  Log of openmx.c:

     5/Oct/2003  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
/*  end stat section */
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#include "tran_prototypes.h"
#include "tran_variables.h"
#include "Inputtools.h"
#include <stdbool.h>
#include <complex.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <math.h>

void Make_VNA_Grid();
void set_O_nm(double ****O);
void output_O_nm(double ****O, double ****OLP, double *****Hks);
void output_atomic_orbitals();
static void Print_CubeTitle(FILE *fp, int EigenValue_flag, double EigenValue);
static void Print_CubeCData_MO(FILE *fp, dcomplex *data, char *op);
static void Set_initial_Hamiltonian(char *mode,
                             int SCF_iter,
                             int Cnt_kind,
                             double *****H0,
                             double *****HNL,
                             double *****H);
#ifdef kcomp
double ****Allocate4D_double(int size_1, int size_2, int size_3, int size_4);
void Free4D_double(double ****buffer);
#else
static inline double ****Allocate4D_double(int size_1, int size_2, int size_3, int size_4);
void Free4D_double(double ****buffer);
#endif
void Free_truncation(int CpyN, int TN, int Free_switch);
int Set_Periodic(int CpyN, int Allocate_switch);
void Set_Hlr();
void Calc_MatrixElements_Vlr(int idx_P);
double Calc_Vlr();
static void Data_Grid_Copy_B2C_XiwenLi(double *data_B, double *data_C);

int main(int argc, char *argv[])
{
  static int numprocs, myid;
  static int MD_iter, i, j, po, ip;
  static char fileMemory[YOUSO10];
  double TStime, TEtime;
  //Added by Yang Zhong
  int postprocess;

  MPI_Comm mpi_comm_parent;

  /* MPI initialize */

  mpi_comm_level1 = MPI_COMM_WORLD;
  MPI_COMM_WORLD1 = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD1, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1, &myid);
  NUMPROCS_MPI_COMM_WORLD = numprocs;
  MYID_MPI_COMM_WORLD = myid;
  Num_Procs = numprocs;

  /* check if OpenMX was called by MPI_spawn. */

  MPI_Comm_get_parent(&mpi_comm_parent);
  if (mpi_comm_parent != MPI_COMM_NULL)
    MPI_spawn_flag = 1;
  else
    MPI_spawn_flag = 0;

  /* for measuring elapsed time */

  dtime(&TStime);

  /* check argv */

  if (argc == 1)
  {

    if (myid == Host_ID)
      printf("\nCould not find an input file.\n\n");
    MPI_Finalize();
    exit(0);
  }

  /* initialize Runtest_flag */

  Runtest_flag = 0;

  /****************************************************
    ./openmx -nt #

    specifies the number of threads in parallelization
    by OpenMP
  ****************************************************/

  openmp_threads_num = 1; /* default */

  po = 0;
  if (myid == Host_ID)
  {
    for (i = 0; i < argc; i++)
    {
      if (strcmp(argv[i], "-nt") == 0)
      {
        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if ((argc - 1) < (ip + 1))
  {
    if (myid == Host_ID)
    {
      printf("cannot find the number of threads\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (po == 1)
  {
    openmp_threads_num = atoi(argv[ip + 1]);

    if (openmp_threads_num <= 0)
    {
      if (myid == Host_ID)
      {
        printf("check the number of threads\n");
      }
      MPI_Finalize();
      exit(0);
    }
  }

  omp_set_num_threads(openmp_threads_num);

  if (myid == Host_ID)
  {
    printf("\nThe number of threads in each node for OpenMP parallelization is %d.\n\n", openmp_threads_num);
  }

  /****************************************************
    ./openmx -show directory

    showing PAOs and VPS used in input files stored in
    "directory".
  ****************************************************/

  if (strcmp(argv[1], "-show") == 0)
  {
    Show_DFT_DATA(argv);
    exit(0);
  }

  /****************************************************
    ./openmx -maketest

    making of *.out files in order to check whether
    OpenMX normally runs on many platforms or not.
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketest") == 0)
  {
    Maketest("S", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtest

   check whether OpenMX normally runs on many platforms
   or not by comparing the stored *.out and generated
   *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1], "-runtest") == 0)
  {
    Runtest("S", argc, argv);
  }

  /****************************************************
   ./openmx -maketestL

    making of *.out files in order to check whether
    OpenMX normally runs for relatively large systems
    on many platforms or not
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestL") == 0)
  {
    Maketest("L", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestL2

    making of *.out files in order to check whether
    OpenMX normally runs for large systems on many
    platforms or not
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestL2") == 0)
  {
    Maketest("L2", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestL3

    making of *.out files in order to check whether
    OpenMX normally runs for large systems on many
    platforms or not
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestL3") == 0)
  {
    Maketest("L3", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestL

   check whether OpenMX normally runs for relatively
   large systems on many platforms or not by comparing
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1], "-runtestL") == 0)
  {
    Runtest("L", argc, argv);
  }

  /****************************************************
   ./openmx -runtestL2

   check whether OpenMX normally runs for large systems
   on many platforms or not by comparing the stored *.out
   and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1], "-runtestL2") == 0)
  {
    Runtest("L2", argc, argv);
  }

  /****************************************************
   ./openmx -runtestL3

   check whether OpenMX normally runs for small, medium size,
   and large systems on many platforms or not by comparing
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1], "-runtestL3") == 0)
  {
    Runtest("L3", argc, argv);
  }

  /*******************************************************
   check memory leak by monitoring actual used memory size
  *******************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-mltest") == 0)
  {
    Memory_Leak_test(argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestG

    making of *.out files in order to check whether
    OpenMX normally runs for geometry optimization
    on many platforms or not
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestG") == 0)
  {
    Maketest("G", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestC

    making of *.out files in order to check whether
    OpenMX normally runs for geometry optimization
    on many platforms or not
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestC") == 0)
  {
    Maketest("C", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestG

   check whether OpenMX normally runs for geometry
   optimization on many platforms or not by comparing
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1], "-runtestG") == 0)
  {
    Runtest("G", argc, argv);
  }

  /****************************************************
   ./openmx -runtestC

   check whether OpenMX normally runs for simultaneous
   optimization for cell and geometry on many platforms
   or not by comparing the stored *.out and generated
   *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1], "-runtestC") == 0)
  {
    Runtest("C", argc, argv);
  }

  /****************************************************
   ./openmx -maketestWF

    making of *.out files in order to check whether
    OpenMX normally runs for generation of Wannier
    functions on many platforms or not
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestWF") == 0)
  {
    Maketest("WF", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestWF

   check whether OpenMX normally runs for generating
   Wannier functions on many platforms or not by comparing
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1], "-runtestWF") == 0)
  {
    Runtest("WF", argc, argv);
  }

  /****************************************************
   ./openmx -maketestNEGF

    making of *.out files in order to check whether
    OpenMX normally runs for NEGF calculations
    on many platforms or not
  ****************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestNEGF") == 0)
  {
    Maketest("NEGF", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestNEGF

    check whether OpenMX normally runs for NEGF calculations
    on many platforms or not
  ****************************************************/

  if (strcmp(argv[1], "-runtestNEGF") == 0)
  {
    Runtest("NEGF", argc, argv);
    MPI_Finalize();
    exit(0);
  }

  /********************************************************************
   ./openmx -maketestCDDF

    making of *.df_re and *.df_im files in order to check whether
    OpenMX normally runs for CDDF calculations on many platforms or not
  *********************************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestCDDF") == 0)
  {
    Maketest("CDDF", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestCDDF

    check whether OpenMX normally runs for CDDF calculations
    on many platforms or not
  ****************************************************/

  if (strcmp(argv[1], "-runtestCDDF") == 0)
  {
    Runtest("CDDF", argc, argv);
    MPI_Finalize();
    exit(0);
  }

  /********************************************************************
   ./openmx -maketestDCLNO

    making of *.out files in order to check whether
    OpenMX normally runs for DC-LNO calculations
    on many platforms or not
  *********************************************************************/

  if ((argc == 2 || argc == 3) && strcmp(argv[1], "-maketestDCLNO") == 0)
  {
    Maketest("DCLNO", argc, argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestDCLNO

   check whether OpenMX normally runs for DCLNO calculations
   on many platforms or not
  ****************************************************/

  if (strcmp(argv[1], "-runtestDCLNO") == 0)
  {
    Runtest("DCLNO", argc, argv);
    MPI_Finalize();
    exit(0);
  }

  /*******************************************************
   check consistency between analytic and numerical forces
  *******************************************************/

  if ((argc == 3 || argc == 4) && strcmp(argv[1], "-forcetest") == 0)
  {

    if (strcmp(argv[2], "0") == 0)
      force_flag = 0;
    else if (strcmp(argv[2], "1") == 0)
      force_flag = 1;
    else if (strcmp(argv[2], "2") == 0)
      force_flag = 2;
    else if (strcmp(argv[2], "3") == 0)
      force_flag = 3;
    else if (strcmp(argv[2], "4") == 0)
      force_flag = 4;
    else if (strcmp(argv[2], "5") == 0)
      force_flag = 5;
    else if (strcmp(argv[2], "6") == 0)
      force_flag = 6;
    else if (strcmp(argv[2], "7") == 0)
      force_flag = 7;
    else if (strcmp(argv[2], "8") == 0)
      force_flag = 8;
    else
    {
      printf("unsupported flag for -forcetest\n");
      exit(0);
    }

    Force_test(argc, argv);
    exit(0);
  }

  /*********************************************************
   check consistency between analytic and numerical stress
  *********************************************************/

  if ((argc == 3 || argc == 4) && strcmp(argv[1], "-stresstest") == 0)
  {

    if (strcmp(argv[2], "0") == 0)
      stress_flag = 0;
    else if (strcmp(argv[2], "1") == 0)
      stress_flag = 1;
    else if (strcmp(argv[2], "2") == 0)
      stress_flag = 2;
    else if (strcmp(argv[2], "3") == 0)
      stress_flag = 3;
    else if (strcmp(argv[2], "4") == 0)
      stress_flag = 4;
    else if (strcmp(argv[2], "5") == 0)
      stress_flag = 5;
    else if (strcmp(argv[2], "6") == 0)
      stress_flag = 6;
    else if (strcmp(argv[2], "7") == 0)
      stress_flag = 7;
    else if (strcmp(argv[2], "8") == 0)
      stress_flag = 8;
    else
    {
      printf("unsupported flag for -stresstest\n");
      exit(0);
    }

    Stress_test(argc, argv);
    MPI_Finalize();
    exit(0);
  }

  /*******************************************************
    check the NEB calculation or not, and if yes, go to
    the NEB calculation.
  *******************************************************/

  if (neb_check(argv))
    neb(argc, argv);

  /*******************************************************
   allocation of CompTime and show the greeting message
  *******************************************************/

  CompTime = (double **)malloc(sizeof(double *) * numprocs);
  for (i = 0; i < numprocs; i++)
  {
    CompTime[i] = (double *)malloc(sizeof(double) * 30);
    for (j = 0; j < 30; j++)
      CompTime[i][j] = 0.0;
  }

  if (myid == Host_ID)
  {
    printf("\n*******************************************************\n");
    printf("*******************************************************\n");
    printf(" Welcome to OpenMX   Ver. %s                           \n", Version_OpenMX);
    printf(" Copyright (C), 2002-2019, T. Ozaki                    \n");
    printf(" OpenMX comes with ABSOLUTELY NO WARRANTY.             \n");
    printf(" This is free software, and you are welcome to         \n");
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("*******************************************************\n");
    printf("*******************************************************\n\n");
  }

  Init_List_YOUSO();
  remake_headfile = 0;
  ScaleSize = 1.2;

  /****************************************************
                   Read the input file
  ****************************************************/

  init_alloc_first();

  CompTime[myid][1] = readfile(argv);
  MPI_Barrier(MPI_COMM_WORLD1);

  /* initialize PrintMemory routine */

  sprintf(fileMemory, "%s%s.memory%i", filepath, filename, myid);
  PrintMemory(fileMemory, 0, "init");
  PrintMemory_Fix();

  /* initialize */

  init();

  /* for DFTD-vdW by okuno */
  /* for version_dftD by Ellner*/
  if (dftD_switch == 1 && version_dftD == 2)
    DFTDvdW_init();
  if (dftD_switch == 1 && version_dftD == 3)
    DFTD3vdW_init();

  /* check "-mltest2" mode */

  po = 0;
  if (myid == Host_ID)
  {
    for (i = 0; i < argc; i++)
    {
      if (strcmp(argv[i], "-mltest2") == 0)
      {
        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if (po == 1)
    ML_flag = 1;
  else
    ML_flag = 0;

  /* check "-forcetest2" mode */

  po = 0;
  if (myid == Host_ID)
  {
    for (i = 0; i < argc; i++)
    {
      if (strcmp(argv[i], "-forcetest2") == 0)
      {
        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if (po == 1)
  {
    force_flag = atoi(argv[ip + 1]);
    ForceConsistency_flag = 1;
  }

  /* check force consistency
     the number of processes
     should be less than 2.
  */

  if (ForceConsistency_flag == 1)
  {

    Check_Force(argv);
    CompTime[myid][20] = OutData(argv[1]);
    Merge_LogFile(argv[1]);
    Free_Arrays(0);
    MPI_Finalize();
    exit(0);
    return 0;
  }

  /* check "-stresstest2" mode */

  po = 0;
  if (myid == Host_ID)
  {

    for (i = 0; i < argc; i++)
    {
      if (strcmp(argv[i], "-stresstest2") == 0)
      {

        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if (po == 1)
  {
    stress_flag = atoi(argv[ip + 1]);
    StressConsistency_flag = 1;
  }

  /* check stress consistency
     the number of processes
     should be less than 2.
  */

  if (StressConsistency_flag == 1)
  {
    Check_Stress(argv);
    CompTime[myid][20] = OutData(argv[1]);
    Merge_LogFile(argv[1]);
    Free_Arrays(0);
    MPI_Finalize();
    exit(0);
    return 0;
  }

  /****************************************************
      SCF-DFT calculations, MD and geometrical
      optimization.
  ****************************************************/

  MD_iter = 1;
  Temp_MD_iter = 1;

  do
  {

    if (MD_switch == 12)
      CompTime[myid][2] += truncation(1, 1); /* EvsLC */
    else if (MD_cellopt_flag == 1)
      CompTime[myid][2] += truncation(1, 1); /* cell optimization */
    else
      CompTime[myid][2] += truncation(MD_iter, 1);

    if (ML_flag == 1 && myid == Host_ID)
      Get_VSZ(MD_iter);

    if (Solver == 4)
    {
      TRAN_Calc_GridBound(mpi_comm_level1, atomnum, WhatSpecies, Spe_Atom_Cut1,
                          Ngrid1, Grid_Origin, Gxyz, tv, gtv, rgtv, Left_tv, Right_tv);

      /* output: TRAN_region[], TRAN_grid_bound */
    }

    if (Solver != 4 || TRAN_SCF_skip == 0)
    {

      input_open(argv[1]);
      input_int("postprocess", &postprocess, 1);
      input_close();

      if (postprocess == 1) // calculate overlap matrices,added by Yang Zhong
      {
        if (myid == Host_ID)
          printf("\n Calculate S, H0 and Hlr ...\n");
        CompTime[myid][3] += DFT(MD_iter, (MD_iter - 1) % orbitalOpt_per_MDIter + 1);
        Set_initial_Hamiltonian("stdout", 1, 0, H0, HNL, H);
        Set_Orbitals_Grid(0);
        Calc_Vlr();
        Set_Hlr();
        if (HS_fileout == 1)
          SCF2File("write", argv[1]);
        if (myid == Host_ID)
        {
          printf("\n Finish calculating S & H0 & Hlr\n");           
        }      
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
      }
      else
      {
        if (myid == Host_ID)
        {
          printf("Wrong postprocess number!\n");
        }
        MPI_Finalize();
        exit(0);
      }

      iterout(MD_iter + MD_Current_Iter, MD_TimeStep * (MD_iter + MD_Current_Iter - 1), filepath, filename);

      /* MD or geometry optimization */
      if (ML_flag == 0)
        CompTime[myid][4] += MD_pac(MD_iter, argv[1]);
    }
    else
    {
      MD_Opt_OK = 1;
    }

    MD_iter++;
    Temp_MD_iter++;

  } while (MD_Opt_OK == 0 && (MD_iter + MD_Current_Iter) <= MD_IterNumber);

  if (TRAN_output_hks)
  {
    /* left is dummy */
    TRAN_RestartFile(mpi_comm_level1, "write", "left", filepath, TRAN_hksoutfilename);
  }

  /****************************************************
               calculate Voronoi charge
  ****************************************************/

  if (Voronoi_Charge_flag == 1)
    Voronoi_Charge();

  /****************************************************
        calculate Voronoi orbital magnetic moment
  ****************************************************/

  if (Voronoi_OrbM_flag == 1)
    Voronoi_Orbital_Moment();

  /****************************************************
        output analysis of decomposed energies
  ****************************************************/

  if (Energy_Decomposition_flag == 1)
    Output_Energy_Decomposition();

  /****************************************************
  making of a file *.frac for the fractional coordinates
  ****************************************************/

  Make_FracCoord(argv[1]);

  /****************************************************
   generate Wannier functions added by Hongming Weng
  ****************************************************/

  /* hmweng */
  if (Wannier_Func_Calc)
  {
    if (myid == Host_ID)
      printf("Calling Generate_Wannier...\n");
    fflush(0);

    Generate_Wannier(argv[1]);
  }

  /****************************************************
      population analysis based on atomic orbitals
              resembling Wannier functions
  ****************************************************/

  if (pop_anal_aow_flag)
  {
    if (myid == Host_ID)
      printf("Population analysis based on atomic orbitals resembling Wannier functions\n");
    fflush(0);

    Population_Analysis_Wannier2(argv);
  }

  /*********************************************************
     Electronic transport calculations based on NEGF:
     transmission, current, eigen channel analysis, and
     real space analysis of current
  *********************************************************/

  if (Solver == 4 && TRAN_analysis == 1)
  {

    /* if SCF is skipped, calculate values of basis functions on each grid */
    if (1 <= TRAN_SCF_skip)
      i = Set_Orbitals_Grid(0);

    if (SpinP_switch == 3)
    {
      TRAN_Main_Analysis_NC(mpi_comm_level1, argc, argv, Matomnum, M2G,
                            GridN_Atom, GridListAtom, CellListAtom,
                            Orbs_Grid, TNumGrid);
    }
    else
    {
      TRAN_Main_Analysis(mpi_comm_level1, argc, argv, Matomnum, M2G,
                         GridN_Atom, GridListAtom, CellListAtom,
                         Orbs_Grid, TNumGrid);
    }
  }

  /*********************************************************
   calculations of core level spectra:

   CLE_Type =-1; NONE
   CLE_Type = 0; XANES0: single particle calculation
   CLE_Type = 1; XANES1: core excitation by delta-SCF
   CLE_Type = 2; XANES2: core excitation with a valence
                         excitation by delta-SCF
  **********************************************************/

  /* calc. of overlap matrix with the position operator */

  if (0 <= CLE_Type)
  {
    Set_OLP_p(OLP_p);

    /* XANES:  single particle calculation */
    if (CLE_Type == 0)
    {
      /* XANES0(); */
    }
  }

  /****************************************************
                  Making of output files
  ****************************************************/

  if (OutData_bin_flag)
    CompTime[myid][20] = OutData_Binary(argv[1]);
  else
    CompTime[myid][20] = OutData(argv[1]);

  /****************************************************
    write connectivity, Hamiltonian, overlap, density
    matrices, and etc. to a file, filename.scfout
  ****************************************************/

  if (HS_fileout == 1)
    SCF2File("write", argv[1]);

  /* elapsed time */

  dtime(&TEtime);
  CompTime[myid][0] = TEtime - TStime;
  Output_CompTime();
  for (i = 0; i < numprocs; i++)
  {
    free(CompTime[i]);
  }
  free(CompTime);

  /* merge log files */
  Merge_LogFile(argv[1]);

  /* free arrays for NEGF */

  if (Solver == 4)
  {

    TRAN_Deallocate_Atoms();
    TRAN_Deallocate_RestartFile("left");
    TRAN_Deallocate_RestartFile("right");
  }

  /* free arrays */

  Free_Arrays(0);

  /* print memory */

  PrintMemory("total", 0, "sum");

  MPI_Barrier(MPI_COMM_WORLD1);
  if (myid == Host_ID)
  {
    printf("\nThe calculation was normally finished.\n");
    fflush(stdout);
  }

  /* if OpenMX is called by MPI_spawn. */

  if (MPI_spawn_flag == 1)
  {

    MPI_Comm_get_parent(&mpi_comm_parent);

    if (mpi_comm_parent != MPI_COMM_NULL)
    {
      // MPI_Comm_disconnect(&mpi_comm_parent);
    }

    fclose(MPI_spawn_stream);

    MPI_Finalize();
  }
  else
  {
    MPI_Finalize();
    exit(0);
  }

  return 0;
}


// added by Xiwen Li
void Set_Hlr(){  
  int Mc_AN;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Calc_MatrixElements_Vlr(Mc_AN);
  }
}

void Calc_MatrixElements_Vlr(int idx_P){
  // calculate < | Vlr[idx_P] | >
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int Nh0,Nh1,Nh2,Nh3;
  int Nc0,Nc1,Nc2,Nc3;
  int MN0,MN1,MN2,MN3;
  int Nloop,OneD_Nloop;
  int *OneD2spin,*OneD2Mc_AN,*OneD2h_AN;
  int numprocs,myid;
  double time0,time1,time2,mflops;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* one-dimensionalization of loops */

  Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Nloop++;
    }
  }

  OneD2Mc_AN = (int*)malloc(sizeof(int)*Nloop);
  OneD2h_AN = (int*)malloc(sizeof(int)*Nloop);

  Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      OneD2Mc_AN[Nloop] = Mc_AN;
      OneD2h_AN[Nloop] = h_AN;
      Nloop++;
    }
  }

  OneD_Nloop = Nloop;

  /* numerical integration */

#pragma omp parallel 
  {
    int Nloop,spin,Mc_AN,h_AN,Gh_AN,Mh_AN,Hwan,NOLG;
    int Gc_AN,Cwan,NO0,NO1,spin0=-1,Mc_AN0=0;
    int i,j,Nc,MN,GNA,Nog,Nh,OMPID,Nthrds;
    int M,N,K,lda,ldb,ldc,ii,jj;
    double alpha,beta,Vpot;
    double sum0,sum1,sum2,sum3,sum4;
    double *ChiV0,*Chi1,*ChiV0_2,*C;

    /* allocation of arrays */

    /* AITUNE */
    double **AI_tmpH;
    {
      /* get size of temporary buffer */
      int AI_MaxNO = 0;
      int spe;
      for(spe = 0; spe < SpeciesNum; spe++){
        if(AI_MaxNO < Spe_Total_NO[spe]){ AI_MaxNO = Spe_Total_NO[spe];}
      }
		
      int spin;

      AI_tmpH = (double**)malloc(sizeof(double*)*AI_MaxNO);

      int i;
      double *p = (double*)malloc(sizeof(double)*AI_MaxNO*AI_MaxNO);
      for(i = 0; i < AI_MaxNO; i++){
        AI_tmpH[i] = p;
        p += AI_MaxNO;
      }
    }
    /* AITUNE */

    /* starting of one-dimensionalized loop */

#pragma omp for schedule(static,1)  /* guided */  /* AITUNE */
  for (Nloop = 0; Nloop < OneD_Nloop; Nloop++){ /* AITUNE */
    int Mc_AN = OneD2Mc_AN[Nloop];
    int h_AN = OneD2h_AN[Nloop];
    int Gc_AN = M2G[Mc_AN];    
    int Gh_AN = natn[Gc_AN][h_AN];
    int Mh_AN = F_G2M[Gh_AN];
    int Cwan = WhatSpecies[Gc_AN];
    int Hwan = WhatSpecies[Gh_AN];
    int GNA = GridN_Atom[Gc_AN];
    int NOLG = NumOLG[Mc_AN][h_AN]; 

    int NO0,NO1;
    NO0 = Spe_Total_NO[Cwan];
    NO1 = Spe_Total_NO[Hwan];
    /* quadrature for Hlr[idx_P]ij  */

    /* AITUNE change order of loop */

    /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */
    
    int i;
    for (i=0; i<NO0; i++){
      int j;
      for (j=0; j<NO1; j++){
        AI_tmpH[i][j] = Hlr[idx_P][Mc_AN][h_AN][i][j];
      }
    }

    int Nog;
    for (Nog=0; Nog<NOLG; Nog++){
      int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
      int MN = MGridListAtom[Mc_AN][Nc];
      int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
      
      double AI_tmp_GVVG = GridVol * Vlr_Grid[idx_P][MN];

      if (G2ID[Gh_AN]==myid){
        int i;
        for (i=0; i<NO0; i++){

          double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
          int j;

          for (j=0; j<NO1; j++){
      AI_tmpH[i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
          }		
        }
        
      }else{
        int i;
        for (i=0; i<NO0; i++){
            
          double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
          int j;

          for (j=0; j<NO1; j++){
      AI_tmpH[i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
          }
        }
      }
      
    }/* Nog */
    
    for (i=0; i<NO0; i++){
      int j;
      for (j=0; j<NO1; j++){
        Hlr[idx_P][Mc_AN][h_AN][i][j] = AI_tmpH[i][j];
      }
    }
    /* AITUNE change order of loop */
  } /* Nloop */

    /* freeing of arrays */
    free(AI_tmpH[0]);
    free(AI_tmpH);
  } /* pragma omp parallel */ 

  /* freeing of arrays */

  free(OneD2Mc_AN);
  free(OneD2h_AN);
}

void Set_initial_Hamiltonian(char *mode,
                            int SCF_iter,
                            int Cnt_kind,
                             double *****H0,
                             double *****HNL,
                             double *****H){
  /***************************************************************
    Cnt_kind
      0:  Uncontracted Hamiltonian
      1:  Contracted Hamiltonian
  ***************************************************************/

  int Mc_AN, Gc_AN, Mh_AN, h_AN, Gh_AN;
  int i, j, k, Cwan, Hwan, NO0, NO1, spin, N, NOLG;
  int Nc, Ncs, GNc, GRc, Nog, Nh, MN, XC_P_switch;
  double TStime, TEtime;
  int numprocs, myid;
  double time0, time1, time2, mflops;
  long Num_C0, Num_C1;

  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  if (myid == Host_ID && strcasecmp(mode, "stdout") == 0 && 0 < level_stdout)
  {
    printf("<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...\n");
    fflush(stdout);
  }

  /*****************************************************
          adding H0+HNL+(HCH) to H
  *****************************************************/

  /* spin non-collinear */

  if (SpinP_switch == 3)
  {
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i = 0; i < Spe_Total_NO[Cwan]; i++)
        {
          for (j = 0; j < Spe_Total_NO[Hwan]; j++)
          {

            if (ProExpn_VNA == 0)
            {
              H[0][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] = F_NL_flag * HNL[2][Mc_AN][h_AN][i][j];
              H[3][Mc_AN][h_AN][i][j] = 0.0;
            }
            else
            {
              H[0][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] + F_NL_flag * HNL[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] + F_NL_flag * HNL[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] = F_NL_flag * HNL[2][Mc_AN][h_AN][i][j];
              H[3][Mc_AN][h_AN][i][j] = 0.0;
            }

            /* Effective Hubbard Hamiltonain --- added by MJ */

            if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter)
            {
              H[0][Mc_AN][h_AN][i][j] += H_Hub[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] += H_Hub[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] += H_Hub[2][Mc_AN][h_AN][i][j];
            }

            /* core hole Hamiltonain */

            if (core_hole_state_flag == 1)
            {
              H[0][Mc_AN][h_AN][i][j] += HCH[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] += HCH[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] += HCH[2][Mc_AN][h_AN][i][j];
            }
          }
        }
      }
    }
  }

  /* spin collinear */

  else
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i = 0; i < Spe_Total_NO[Cwan]; i++)
        {
          for (j = 0; j < Spe_Total_NO[Hwan]; j++)
          {
            for (spin = 0; spin <= SpinP_switch; spin++)
            {

              if (ProExpn_VNA == 0)
              {
                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[spin][Mc_AN][h_AN][i][j];
              }
              else
              {
                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] + F_NL_flag * HNL[spin][Mc_AN][h_AN][i][j];
              }

              /* Effective Hubbard Hamiltonain --- added by MJ */
              if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter)
              {
                H[spin][Mc_AN][h_AN][i][j] += H_Hub[spin][Mc_AN][h_AN][i][j];
              }

              /* core hole Hamiltonain */
              if (core_hole_state_flag == 1)
              {
                H[spin][Mc_AN][h_AN][i][j] += HCH[spin][Mc_AN][h_AN][i][j];
              }
            }
          }
        }
      }
    }
  }

  if (Cnt_kind == 1)
  {
    Contract_Hamiltonian(H, CntH, OLP, CntOLP);
    if (SO_switch == 1)
      Contract_iHNL(iHNL, iCntHNL);
  }
}

#ifdef kcomp
double ****Allocate4D_double(int size_1, int size_2, int size_3, int size_4)
#else
static inline double ****Allocate4D_double(int size_1, int size_2, int size_3, int size_4)
#endif
{
  int i, j, k, l;

  double ****buffer = (double ****)malloc(sizeof(double ***) * size_1);
  buffer[0] = (double ***)malloc(sizeof(double **) * size_1 * size_2);
  buffer[0][0] = (double **)malloc(sizeof(double *) * size_1 * size_2 * size_3);
  buffer[0][0][0] = (double *)malloc(sizeof(double) * size_1 * size_2 * size_3 * size_4);

  for (i = 0; i < size_1; i++)
  {
    buffer[i] = buffer[0] + i * size_2;
    for (j = 0; j < size_2; j++)
    {
      buffer[i][j] = buffer[0][0] + (i * size_2 + j) * size_3;
      for (k = 0; k < size_3; k++)
      {
        buffer[i][j][k] = buffer[0][0][0] + ((i * size_2 + j) * size_3 + k) * size_4;
        for (l = 0; l < size_4; l++)
        {
          buffer[i][j][k][l] = 0.0;
        }
      }
    }
  }

  return buffer;
}

static void Print_CubeCData_MO(FILE *fp, dcomplex *data, char *op)
{
  int i1, i2, i3;
  int GN;
  int cmd;
  char buf[fp_bsize]; /* setvbuf */

  setvbuf(fp, buf, _IOFBF, fp_bsize); /* setvbuf */

  if (strcmp(op, "r") == 0)
  {
    cmd = 1;
  }
  else if (strcmp(op, "i") == 0)
  {
    cmd = 2;
  }
  else
  {
    printf("Print_CubeCData: op=%s not supported\n", op);
    return;
  }

  for (i1 = 0; i1 < Ngrid1; i1++)
  {
    for (i2 = 0; i2 < Ngrid2; i2++)
    {
      for (i3 = 0; i3 < Ngrid3; i3++)
      {
        GN = i1 * Ngrid2 * Ngrid3 + i2 * Ngrid3 + i3;
        switch (cmd)
        {
        case 1:
          fprintf(fp, "%13.3E", data[GN].r);
          break;
        case 2:
          fprintf(fp, "%13.3E", data[GN].i);
          break;
        }
        if ((i3 + 1) % 6 == 0)
        {
          fprintf(fp, "\n");
        }
      }
      /* avoid double \n\n when Ngrid3%6 == 0  */
      if (Ngrid3 % 6 != 0)
        fprintf(fp, "\n");
    }
  }
}

static void Print_CubeTitle(FILE *fp, int EigenValue_flag, double EigenValue)
{
  int ct_AN;
  int spe;

  if (EigenValue_flag == 0)
  {
    fprintf(fp, " SYS1\n SYS1\n");
  }
  else
  {
    fprintf(fp, " Absolute eigenvalue=%10.7f (Hartree)  Relative eigenvalue=%10.7f (Hartree)\n",
            EigenValue, EigenValue - ChemP);
    fprintf(fp, " Chemical Potential=%10.7f (Hartree)\n", ChemP);
  }

  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
          atomnum, Grid_Origin[1], Grid_Origin[2], Grid_Origin[3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
          Ngrid1, gtv[1][1], gtv[1][2], gtv[1][3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
          Ngrid2, gtv[2][1], gtv[2][2], gtv[2][3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
          Ngrid3, gtv[3][1], gtv[3][2], gtv[3][3]);

  for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
  {
    spe = WhatSpecies[ct_AN];
    fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf%12.6lf\n",
            Spe_WhatAtom[spe],
            Spe_Core_Charge[spe] - InitN_USpin[ct_AN] - InitN_DSpin[ct_AN],
            Gxyz[ct_AN][1], Gxyz[ct_AN][2], Gxyz[ct_AN][3]);
  }
}

// added by Yang Zhong
double ***Allocate3D_double(int m, int n, int t)
{
  int i = 0;
  int k = 0;
  double ***result = NULL;
  if ((m > 0) && (n > 0) && (t > 0))
  {
    double **pp = NULL;
    double *p = NULL;
    result = (double ***)malloc(m * sizeof(double **)); // key
    pp = (double **)malloc(m * n * sizeof(double *));   // key
    p = (double *)malloc(m * n * t * sizeof(double));   // key
    if ((result != NULL) && (pp != NULL) && (p != NULL))
    {
      for (i = 0; i < m; i++)
      {
        result[i] = pp + i * n; // 三维元素存二维地址
        for (k = 0; k < n; k++)
        {
          result[i][k] = p + k * t; // // 二维元素存一维地址
        }
        p = p + n * t;
      }
    }
    else
    {
      free(result);
      free(pp);
      free(p);
      result = NULL;
      pp = NULL;
      p = NULL;
    }
  }
  return result;
}

void Free3D_double(double ***p)
{
  if (*p != NULL)
  {
    if (**p != NULL)
    {
      free(**p);
      **p = NULL;
    }
    free(*p);
    *p = NULL;
  }
  free(p);
  p = NULL;
}

// added by Xiwen Li
double Calc_Vlr(){
  int k1,k2,k3;
  int N2D,GNs,GN,BN_CB,BN_AB;
  int N3[4];
  double time0,Rc;
  double tmp0,sk1,sk2,sk3;
  double Gx,Gy,Gz,fac_invG2; 
  double TStime,TEtime,etime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

	int idx_P;
	double **ReVlrk, **ImVlrk;
  double KK,sigma,thx,thy,thz;


  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout){
    printf("Calculate Hlr using FFT...\n");
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);
	
	/* allocation of arrays */
  ReVlrk = (double *)malloc(sizeof(double*)*(Matomnum+MatomnumF+MatomnumS+2));
  ImVlrk = (double *)malloc(sizeof(double*)*(Matomnum+MatomnumF+MatomnumS+2));
	for(idx_P=1;idx_P<=(Matomnum+MatomnumF+MatomnumS+1);idx_P++){
		ReVlrk[idx_P] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
		ImVlrk[idx_P] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 	
	}  
	/****************************************************
		  find the long range potential in reciprocal space
	****************************************************/  
  sigma = 1.0;

	/* set ReVlrk and ImVlrk */
  tmp0 = 4.*PI/Cell_Volume;

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;	
	
  for(idx_P=1;idx_P<=(Matomnum+MatomnumF+MatomnumS+1);idx_P++){
    for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

      GN = BN_CB + GNs;     
      k3 = GN/(Ngrid2*Ngrid1);    
      k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
      k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

      if (k1<Ngrid1/2) sk1 = (double)k1;
      else             sk1 = (double)(k1 - Ngrid1);

      if (k2<Ngrid2/2) sk2 = (double)k2;
      else             sk2 = (double)(k2 - Ngrid2);

      if (k3<Ngrid3/2) sk3 = (double)k3;
      else             sk3 = (double)(k3 - Ngrid3);

      Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
      Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
      Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];     

      KK = Gx*Gx + Gy*Gy + Gz*Gz;
      thx = -Gx*Gxyz[idx_P][1];thy = -Gy*Gxyz[idx_P][2];thz = -Gz*Gxyz[idx_P][3];
      if(KK>0){
        ReVlrk[idx_P][BN_CB] = tmp0/KK*exp(-sigma*sigma*KK/2.)*cos(thx+thy+thz);
        ImVlrk[idx_P][BN_CB] = -tmp0/KK*exp(-sigma*sigma*KK/2.)*sin(thx+thy+thz);
      }
      else{
        ReVlrk[idx_P][BN_CB] = 0.0;
        ImVlrk[idx_P][BN_CB] = 0.0;
      }
    }
  }
	
	/****************************************************
		  find the long range potential in real space
	****************************************************/
  for(idx_P=1;idx_P<=(Matomnum+MatomnumF+MatomnumS+1);idx_P++){
    Get_Value_inReal(0,Vlr_Grid_B[idx_P],Vlr_Grid_B[idx_P],ReVlrk[idx_P],ImVlrk[idx_P]);
    // copy from partition B to partition C
    Data_Grid_Copy_B2C_XiwenLi(Vlr_Grid_B[idx_P], Vlr_Grid[idx_P]);    
  }
  
  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}

static void Data_Grid_Copy_B2C_XiwenLi(double *data_B, double *data_C){
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_S[NN_B2C_S]); 
  Work_Array_Rcv_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_R[NN_B2C_R]); 

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2C_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2C_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2C_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2C_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[gp], Num_Rcv_Grid_B2C[IDR],
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }

  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Vlr_Grid_B to Work_Array_Snd_Grid_B2C */

    for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
      BN = Index_Snd_Grid_B2C[IDS][LN];
      Work_Array_Snd_Grid_B2C[gp+LN] = data_B[BN];


    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[gp], Num_Snd_Grid_B2C[IDS], 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){
      gp = GP_B2C_S[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
        CN = Index_Rcv_Grid_B2C[IDR][LN];
        data_C[CN] = Work_Array_Snd_Grid_B2C[gp+LN];
      } /* LN */   

    }
    else {
      gp = GP_B2C_R[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	      CN = Index_Rcv_Grid_B2C[IDR][LN];
        data_C[CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
      }
    }
  }
  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
}

