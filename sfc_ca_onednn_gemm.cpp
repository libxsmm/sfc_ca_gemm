/****************************************************************************
 * Copy of sfc_ca_gemm.cpp using oneDNN brgemm instead of LIBXSMM brgemm. *
 ***************************************************************************/
#include "sfc_ca_gemm.hpp"

template<typename DType>
void run_gemm(gemm_config_t *config, DType *A, DType *B, DType *C) {
  // Unpack configuration struct
  long M = config->M, N = config->N, K = config->K, Mb = config->Mb, Nb = config->Nb, Kb = config->Kb, bm = config->bm, bn = config->bn, bk = config->bk, K_layers = config->K_layers, brcount = config->brcount;
  DType **scratch_C = (DType**)config->gemm_scratch;
  unsigned char *sfc_index_map = config->sfc_index_map;
  unsigned int index_tsize = config->index_tsize;
  libxsmm_meltwfunction_unary zero_kernel = config->zero_kernel;
  libxsmm_tilecfgfunction tileconfig_kernel = config->tileconfig_kernel;
  libxsmm_tilecfgfunction tilerelease_kernel = config->tilerelease_kernel;
  libxsmm_meltwfunction_binary l_add_kernel = config->l_add_kernel;
  libxsmm_meltwfunction_unary l_reduce_kernel = config->l_reduce_kernel;
  // Get oneDNN brgemm kernel from config
  dnnl::ukernel::brgemm* brgemm_onednn = (dnnl::ukernel::brgemm*)config->onednn_brgemm_kernel;
  // Get oneDNN kernel and thread-local data from config
  std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> ***tl_offsets_array = (std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> ***)config->onednn_A_B_offsets_ptrs;
  long Kb_per_layer = (Kb + K_layers - 1) / K_layers;
  long Kb_last_layer = Kb - (K_layers - 1) * Kb_per_layer;

#pragma omp parallel
  {
      int tid = omp_get_thread_num();
      auto &tl_A_B_offsets = *(*tl_offsets_array)[tid];
      float C_acc_buffer[bm * bn];
      unsigned char tl_scratchpad_raw[config->onednn_scratchpad_size];
      dnnl::ukernel::attr_params params;

      if (brgemm_onednn != NULL)  brgemm_onednn->set_hw_context();
      for (int i_k = 0; i_k < Kb_per_layer; i_k += brcount)
      {
#pragma omp for nowait
        for (int i_sfc = 0; i_sfc < Mb * Nb * K_layers; i_sfc++) {
          long brcount_use = brcount;
          libxsmm_gemm_param gemm_param;
          int i_m, i_n, i_k_layer = 0;
          sfc_ca_gemm_extract_indices_from_sfc(&i_m, &i_n, sfc_index_map, i_sfc % (Mb * Nb), index_tsize);
          i_k_layer = i_sfc / (Mb * Nb);
          if (i_k_layer == K_layers - 1) {
            if (i_k + brcount > Kb_last_layer) {
              brcount_use = Kb_last_layer - i_k;
            }
          } else {
            if (i_k + brcount > Kb_per_layer) {
              brcount_use = Kb_per_layer - i_k;
            }
          }
          gemm_param.op.tertiary = (void *)&brcount_use;
          gemm_param.a.primary = (void *)((DType *)A + i_m * K * bm + i_k * bk * bm + i_k_layer * (K / K_layers) * bm);
          gemm_param.b.primary = (void *)((DType *)B + i_n * K * bn + i_k * bk * bn + i_k_layer * (K / K_layers) * bn);
          gemm_param.c.primary = (i_k_layer > 0) ? (void *)((DType *)scratch_C[i_k_layer - 1] + i_n * M * bn + i_m * bn * bm)
                                                 : (void *)((DType *)C + i_n * M * bn + i_m * bn * bm);
          if (i_k == 0) {
              libxsmm_meltw_unary_param zero_param;
              zero_param.out.primary = (void *)gemm_param.c.primary;
              zero_kernel(&zero_param);
          }
          // Prepare post-ops: pointer to destination for binary add
          const void* binary_po_ptr = gemm_param.c.primary;
          params.set_post_ops_args(&binary_po_ptr);
          // Execute oneDNN brgemm
          brgemm_onednn->execute(gemm_param.b.primary, gemm_param.a.primary, tl_A_B_offsets, 
                                 C_acc_buffer, gemm_param.c.primary, tl_scratchpad_raw, params);
      }
    }
    if (brgemm_onednn != NULL) dnnl::ukernel::brgemm::release_hw_context();
  }
  if (K_layers > 1) {
    #pragma omp parallel for
    for (int i_red = 0; i_red < Mb*Nb; i_red++) {
      int i_m, i_n;
      sfc_ca_gemm_extract_indices_from_sfc(&i_m, &i_n, sfc_index_map, i_red, index_tsize);
      if (K_layers == 2) {
        libxsmm_meltw_binary_param add_param;
        add_param.in0.primary  = (void*)((DType*)scratch_C[0] + i_n * M * bn + i_m * bn * bm );
        add_param.in1.primary  = (void*)((DType*)C + i_n * M * bn + i_m * bn * bm );       
        add_param.out.primary = (void*)((DType*)C + i_n * M * bn + i_m * bn * bm );
        l_add_kernel(&add_param);
      } else {
        libxsmm_meltw_binary_param add_param;
        libxsmm_meltw_unary_param reduce_param;
        DType reduce_scratch[bm*bn];
        reduce_param.in.primary = (void*)((DType*)scratch_C[0] + i_n * M * bn + i_m * bn * bm );
        reduce_param.out.primary  = (void*)reduce_scratch;
        add_param.in0.primary  = (void*)reduce_scratch;
        add_param.in1.primary  = (void*)((DType*)C + i_n * M * bn + i_m * bn * bm );       
        add_param.out.primary = (void*)((DType*)C + i_n * M * bn + i_m * bn * bm );
        l_reduce_kernel(&reduce_param);
        l_add_kernel(&add_param);
      }
    }
  }
  return;
}

template<typename DType>
void run_gemm_n_layers(long n_layers, gemm_config_t *config, DType **A, DType **BC) {
  for (int i = 0; i < n_layers; i++) {
    run_gemm<DType>(config, A[i], BC[2*i], BC[2*i+1]);
  }
  return;
}

template<typename DType>
int gemm_benchmark(int argc, char** argv) {
  // Setup default GEMM sizes
  long M = 1024*4, N = 1024*4, K = 1024*4;
  long bm = 32, bn = 32, bk = 32;
  long kbf = 1;
  long K_layers = 1;
  long n_layers = 1;
  long n_iters = 1;
  long i;
  long check_correctness = 0;
  long m_step = 1, n_step = 1;

  ifreq = 1.0 / getFreq();
  // Read command line arguments
  if (argc > 1) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    bm = atoi(argv[4]);
    while (M % bm != 0) bm--;
    bn = atoi(argv[5]);
    while (N % bn != 0) bn--;
    bk = atoi(argv[6]);
    while (K % bk != 0 || bk % 2 != 0) {
      bk--;
      if (bk <= 0)
      {
        printf("Error: could not find suitable bk value\n");
        exit(0);
      }
    }
    if (argc > 7)
    {
      kbf = atoi(argv[7]);
    }
    if (argc > 8){
      K_layers = atoi(argv[8]);
    }
    if (argc > 9) {
      n_layers = atoi(argv[9]);
      if (n_layers == -1) {
        double size_total = (double)sizeof(DType)*(double)1.0*((double)M*(double)K +(double)M*(double)N +(double)K*(double)N)/(1024.0*1024.0*1024.0);
        double low_limit_in_gb = 5.0;
        n_layers = 1;
        while (size_total < low_limit_in_gb) {
          n_layers++;
          size_total = (double)sizeof(DType)*(double)n_layers*((double)M*(double)K +(double)M*(double)N +(double)K*(double)N)/(1024.0*1024.0*1024.0);
        }
        printf("Autocalculated %ld layers with total size %.2g GB\n", n_layers, size_total);
      }
    }
    if (argc > 10) {
      n_iters = atoi(argv[10]);
    }
    if (argc > 11) {
      check_correctness = atoi(argv[11]);
    }
    // argv[12] is dtype, parsed in main()
    if (argc > 13) {
      m_step = atoi(argv[13]);
    }
    if (argc > 14) {
      n_step = atoi(argv[14]);
    }
  }
  
  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  // Allocate buffers
  DType **BC = (DType**) malloc((2*n_layers)*sizeof(DType*));
  check_null_ptr(BC, "BC array");
  DType **A = (DType**) malloc(n_layers    *sizeof(DType*));
  check_null_ptr(A, "A array");
  for (i = 0; i < n_layers; i++) {
    A[i] = (DType*) libxsmm_aligned_malloc(M*K*sizeof(DType), ALIGNMENT_SIZE);
  }
  for (i = 0; i < 2*n_layers; i++) {
    if (i%2 == 0) {
      // Allocate buffer for B
      BC[i] = (DType*) libxsmm_aligned_malloc(K*N*sizeof(DType), ALIGNMENT_SIZE);
    } else {
      // Allocate buffer for C
      BC[i] = (DType*) libxsmm_aligned_malloc(M*N*sizeof(DType), ALIGNMENT_SIZE);
    }
    check_null_ptr(BC[i], "BC[i] array"); 
  }
  
  // Allocate reference buffers
  float *naive_b = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_b, "naive_b array");
  float *naive_c = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_c, "naive_c array");
  float *naive_c_opt = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_c_opt, "naive_c_opt array");
  float *naive_a = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_a, "naive_a array");
  DType *naive_b_lp  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_b_lp, "naive_b_lp array");
  DType *naive_c_lp = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_c_lp, "naive_c_lp array");
  DType *naive_a_lp = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_a_lp, "naive_a_lp array");
  
  // Init buffers
  init_buf( naive_b,    K*N, 0, 0 );
  init_buf( naive_c,    M*N, 0, 0 );
  init_buf( naive_a,    M*K, 0, 0 );
  sfc_ca_gemm_rne_convert_fp32_lp<DType>( naive_b,     (void*)naive_b_lp,     N*K);
  sfc_ca_gemm_convert_lp_f32<DType>( (void*)naive_b_lp, naive_b, N*K);
  sfc_ca_gemm_rne_convert_fp32_lp<DType>( naive_c,    (void*)naive_c_lp,    N*M);
  sfc_ca_gemm_convert_lp_f32<DType>( (void*)naive_c_lp, naive_c, N*M);
  sfc_ca_gemm_rne_convert_fp32_lp<DType>( naive_a,    (void*)naive_a_lp,    M*K);
  sfc_ca_gemm_convert_lp_f32<DType>( (void*)naive_a_lp, naive_a, M*K);
  for (i = 0; i < n_layers; i++) {
    sfc_ca_gemm_matrix_copy_KC_to_KCCK<DType>( (void*)naive_a_lp, (void*)A[i], K, M, bk, bm);
    sfc_ca_gemm_matrix_copy_NC_to_NCNC<DType>( (void*)naive_b_lp, (void*)BC[2*i] , N, K, bn, bk);
    sfc_ca_gemm_matrix_copy_NC_to_NCNC<DType>( (void*)naive_c_lp, (void*)BC[2*i+1], N, M, bn, bm);
  }
  
  // Compute reference if requested
  if (check_correctness) {
    naive_gemm_t naive_param;
    naive_param.N = N;
    naive_param.K = K;
    naive_param.M = M;
    naive_param.fuse_type = 0;
    naive_gemm_fp(&naive_param, naive_b, naive_c, naive_a);
    sfc_ca_gemm_rne_convert_fp32_lp<DType>( naive_c,     (void*)naive_c_lp, (N*M) );
    sfc_ca_gemm_convert_lp_f32<DType>( (void*)naive_c_lp, naive_c, N*M);
  } 
  
  // Setup GEMM configuration with oneDNN brgemm
  gemm_config_t *gemm_cfg = setup_gemm_config_onednn<DType>(M, N, K, bm, bn, bk, kbf, K_layers, m_step, n_step);

  // Warmup iteration
  run_gemm_n_layers<DType>(n_layers, gemm_cfg, A, BC);

  // Check correctness if requested
  printf("##############################################################\n");
  printf("    %ld Sets of GEMMS with sizes  %ld x %ld x %ld  (M x N x K)  \n", n_layers, M, N, K);
  printf("##############################################################\n");

  if (check_correctness) {
    libxsmm_matdiff_info norms, diff;
    libxsmm_matdiff_clear(&norms);
    libxsmm_matdiff_clear(&diff);
    sfc_ca_gemm_matrix_copy_NCNC_to_NC<DType>( (void*)BC[2*n_layers-1], (void*)naive_c_lp, N, M, bn, bm );
    sfc_ca_gemm_convert_lp_f32<DType>( (void*)naive_c_lp, naive_c_opt, N*M );
    printf("##########################################\n");
    printf("#           Correctness                  #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*M, 1, naive_c, naive_c_opt, 0, 0);
    printf("L1 reference  : %.25g\n", norms.l1_ref);
    printf("L1 test       : %.25g\n", norms.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms.l2_rel);
    printf("Linf abs.error: %.24f\n", norms.linf_abs);
    printf("Linf rel.error: %.24f\n", norms.linf_rel);
    printf("Check-norm    : %.24f\n", norms.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms);
  }

  // benchmark the GEMM
  auto t_start = getTime();
  for (long it = 0; it < n_iters; it++) {
    run_gemm_n_layers<DType>(n_layers, gemm_cfg, A, BC);
  }
  auto t_end = getTime();

  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Effective model sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*(double)M*(double)K)/(1024.0*1024.0*1024.0));
  printf("Effective total GEMM sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*((double)M*(double)K + (double)M*(double)N + (double)K*(double)N ))/(1024.0*1024.0*1024.0));
  printf("Effective A BW is %.5g GB/s\n", (((double)sizeof(DType)*(double)n_layers*(double)M*(double)K) / (1024.0*1024.0*1024.0))/((t_end-t_start)/(1.0*n_iters)));
  printf("MEASURE %.5g SFC_CA_GEMM_%ld_%ld_%ld_%ld_%ld_%ld_bf%ld_replication_%ld_threads%d\n", gflop / ((t_end - t_start) / (1.0 * n_iters)), M, N, K, bm, bn, bk, kbf, K_layers, omp_get_max_threads());

  // Free buffers
  libxsmm_free(naive_b);
  libxsmm_free(naive_c);
  libxsmm_free(naive_c_opt);
  libxsmm_free(naive_a);
  libxsmm_free(naive_b_lp);
  libxsmm_free(naive_c_lp);
  libxsmm_free(naive_a_lp);
  for (i = 0; i < n_layers; i++) {
    libxsmm_free(A[i]);
  }
  for (i = 0; i < 2*n_layers; i++) {
    libxsmm_free(BC[i]);
  }
  // Free config buffers
  if (gemm_cfg->sfc_index_map != NULL) {
    libxsmm_free(gemm_cfg->sfc_index_map);
  }

  DType **output_partial_array = (DType**)gemm_cfg->gemm_scratch;
  if (output_partial_array[0] != NULL)
  {
    libxsmm_free(output_partial_array[0]);
  }
  
  // Free oneDNN-specific resources
  if (gemm_cfg->onednn_brgemm_kernel != NULL) {
    dnnl::ukernel::brgemm* brgemm_onednn = (dnnl::ukernel::brgemm*)gemm_cfg->onednn_brgemm_kernel;
    delete brgemm_onednn;
  }
  if (gemm_cfg->onednn_A_B_offsets_ptrs != NULL) {
    std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>>*** tl_offsets_array = 
      (std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>>***)gemm_cfg->onednn_A_B_offsets_ptrs;
    int num_threads = omp_get_max_threads();
    for (int t = 0; t < num_threads; t++) {
      delete (*tl_offsets_array)[t];
    }
    delete[] (*tl_offsets_array);
    delete[] tl_offsets_array;
  }
  
  free(BC);
  free(A);
  libxsmm_free(gemm_cfg->gemm_scratch);
  delete gemm_cfg;
  return 0;
}

int main(int argc, char** argv) {
  int use_dtype = 1;
  if (argc > 12) {
    if (strcmp(argv[12],"BF16") == 0) {
      use_dtype = 1;
    }
    if (strcmp(argv[12],"BF8") == 0) {
      use_dtype = 2;
    }
    if (strcmp(argv[12],"FP32") == 0) {
      use_dtype = 3;
    }
  }
  if (use_dtype == 1) {
    return gemm_benchmark<libxsmm_bfloat16>(argc, argv);  
  } else if (use_dtype == 2) {
    return gemm_benchmark<libxsmm_bfloat8>(argc, argv);
  } else if (use_dtype == 3) {
    return gemm_benchmark<float>(argc, argv);
  } else {
    return 0;
  }
}
