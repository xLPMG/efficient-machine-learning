import torch
import triton
import triton.language

@triton.autotune(
    configs = [ triton.Config( { 'SIZE_BLOCK_L': 1,
                                 'SIZE_BLOCK_M': 64,
                                 'SIZE_BLOCK_N': 64,
                                 'SIZE_BLOCK_K': 16 },
                                 num_stages = 3,
                                 num_warps = 2 ) ],
    key = [ 'size_a_l',
            'size_b_l',
            'size_m',
            'size_n',
            'size_k' ]
)
@triton.jit
def triton_bgemm_kernel( ptr_a,
                         ptr_b,
                         ptr_c,
                         size_a_l,
                         size_b_l,
                         size_m,
                         size_n,
                         size_k,
                         stride_a_l,
                         stride_a_m,
                         stride_a_k,
                         stride_b_l,
                         stride_b_k,
                         stride_b_n,
                         stride_c_l,
                         stride_c_m,
                         stride_c_n,
                         SIZE_BLOCK_L: triton.language.constexpr,
                         SIZE_BLOCK_M: triton.language.constexpr,
                         SIZE_BLOCK_N: triton.language.constexpr,
                         SIZE_BLOCK_K: triton.language.constexpr ):
    pid = triton.language.program_id( axis = 0 )

    # derive location of pid
    num_blocks_m = triton.language.cdiv( size_m, SIZE_BLOCK_M )
    num_blocks_n = triton.language.cdiv( size_n, SIZE_BLOCK_N )

    pid_l = pid // ( num_blocks_m * num_blocks_n )
    pid_m = pid % ( num_blocks_m * num_blocks_n )
    pid_m //= num_blocks_n
    pid_n = pid % num_blocks_n

    # A offsets
    offsets_a_m = pid_m * SIZE_BLOCK_M
    offsets_a_m += triton.language.arange( 0, SIZE_BLOCK_M )
    offsets_a_m %= size_m

    # B offsets
    offsets_b_n = pid_n * SIZE_BLOCK_N
    offsets_b_n += triton.language.arange( 0, SIZE_BLOCK_N )
    offsets_b_n %= size_n

    # C offsets
    offsets_c_m = pid_m * SIZE_BLOCK_M
    offsets_c_m += triton.language.arange( 0, SIZE_BLOCK_M )
    offsets_c_n = pid_n * SIZE_BLOCK_N
    offsets_c_n += triton.language.arange( 0, SIZE_BLOCK_N )

    # L and K offsets
    offset_l = pid_l * SIZE_BLOCK_L # L blocking has to fit
    offsets_k = triton.language.arange( 0, SIZE_BLOCK_K )

    for id_l in range( 0, SIZE_BLOCK_L ):
        # block pointers
        block_ptrs_a  = ptr_a
        if size_a_l > 1:
            block_ptrs_a += (offset_l + id_l) * stride_a_l
        block_ptrs_a += (offsets_a_m[:, None] * stride_a_m) \
                      + (offsets_k[None, :]   * stride_a_k)

        block_ptrs_b = ptr_b
        if size_b_l > 1:
            block_ptrs_b += (offset_l + id_l) * stride_b_l
        block_ptrs_b += (offsets_k[:, None]   * stride_b_k) \
                      + (offsets_b_n[None, :] * stride_b_n)

        # accumulator
        accum = triton.language.zeros( ( SIZE_BLOCK_M,
                                        SIZE_BLOCK_N ),
                                    dtype = triton.language.float32 )

        for block_k in range( 0, triton.language.cdiv( size_k, SIZE_BLOCK_K) ):
            mask_a = offsets_k[None, :] < ( size_k - block_k * SIZE_BLOCK_K )
            a = triton.language.load( block_ptrs_a,
                                      mask = mask_a,
                                      other = 0.0 )
            
            mask_b = offsets_k[:, None] < ( size_k - block_k * SIZE_BLOCK_K )
            b = triton.language.load( block_ptrs_b,
                                      mask = mask_b,
                                      other = 0.0 )


            accum = triton.language.dot( a,
                                         b,
                                         accum,
                                         allow_tf32 = False )

            block_ptrs_a += SIZE_BLOCK_K * stride_a_k
            block_ptrs_b += SIZE_BLOCK_K * stride_b_k

        block_ptrs_c = ptr_c
        block_ptrs_c += (offset_l + id_l) * stride_c_l
        block_ptrs_c += (offsets_c_m[:, None] * stride_c_m) \
                      + (offsets_c_n[None, :] * stride_c_n)

        mask_c  = offsets_c_m[:, None] < size_m
        mask_c &= offsets_c_n[None, :] < size_n
        triton.language.store( block_ptrs_c,
                               accum,
                               mask = mask_c )

def triton_bgemm( a,
                  b ):
    size_l = max( a.size( 0 ), b.size( 0 ) )
    size_m = a.size( 1 )
    size_n = b.size( 2 )
    dtype  = a.dtype

    c = torch.empty( ( size_l,
                       size_m,
                       size_n ),
                     device = 'cuda',
                     dtype = dtype )
    
    grid  = lambda META: (   triton.cdiv(size_l, META['SIZE_BLOCK_L']) \
                           * triton.cdiv(size_m, META['SIZE_BLOCK_M']) \
                           * triton.cdiv(size_n, META['SIZE_BLOCK_N']), )

    triton_bgemm_kernel[ grid ]( ptr_a         = a,
                                 ptr_b         = b,
                                 ptr_c         = c,
                                 size_a_l      = a.size( 0 ),
                                 size_b_l      = b.size( 0 ),
                                 size_m        = a.size( 1 ),
                                 size_n        = b.size( 2 ),
                                 size_k        = a.size( 2 ),
                                 stride_a_l    = a.stride( 0 ),
                                 stride_a_m    = a.stride( 1 ),
                                 stride_a_k    = a.stride( 2 ),
                                 stride_b_l    = b.stride( 0 ),
                                 stride_b_k    = b.stride( 1 ),
                                 stride_b_n    = b.stride( 2 ),
                                 stride_c_l    = c.stride( 0 ),
                                 stride_c_m    = c.stride( 1 ),
                                 stride_c_n    = c.stride( 2 ) )
    
    return c

if __name__ == "__main__":
    # MLP-Mixer B/16
    mixer_config = { 'hidden_size_c':   768,
                     'seq_len_s':       196,
                     'mlp_dim_dc':      3072,
                     'mlp_dim_ds':      383,
                     'batch_size':      64,
                     'dtype':           torch.float32 }

    # set up matrix multiplication sizes
    matmul_mlp_tokens_0 = { 'L': mixer_config['batch_size'],
                            'M': mixer_config['mlp_dim_ds'],
                            'N': mixer_config['hidden_size_c'],
                            'K': mixer_config['seq_len_s'] }
    
    matmul_mlp_tokens_1 = { 'L': mixer_config['batch_size'],
                            'M': mixer_config['seq_len_s'],
                            'N': mixer_config['hidden_size_c'],
                            'K': mixer_config['mlp_dim_ds'] }
    
    matmul_mlp_channels_0 = { 'L': mixer_config['batch_size'],
                              'M': mixer_config['seq_len_s'],
                              'N': mixer_config['mlp_dim_dc'],
                              'K': mixer_config['hidden_size_c'] }
    
    matmul_mlp_channels_1 = { 'L': mixer_config['batch_size'],
                              'M': mixer_config['seq_len_s'],
                              'N': mixer_config['hidden_size_c'],
                              'K': mixer_config['mlp_dim_dc'] }

    # allocate tensors
    a = torch.randn( 1,
                     matmul_mlp_channels_0['M'],
                     matmul_mlp_channels_0['K'],
                     device = 'cuda',
                     dtype = mixer_config['dtype'] )
    
    b = torch.randn( matmul_mlp_channels_0['L'],
                     matmul_mlp_channels_0['K'],
                     matmul_mlp_channels_0['N'],
                     device = 'cuda',
                     dtype = mixer_config['dtype'] )
    
    c = torch.empty( matmul_mlp_channels_0['L'],
                     matmul_mlp_channels_0['M'],
                     matmul_mlp_channels_0['N'],
                     device = 'cuda',
                     dtype = mixer_config['dtype'] )

    c_triton = triton_bgemm( a, b )
    c_torch = torch.matmul( a, b )

    print( 'max abs value of triton bgemm:' )
    print( '  ', torch.max( torch.abs( c_triton ) ) )
    print( 'max asbs value of torch matmul:' )
    print( '  ', torch.max( torch.abs( c_torch ) ) )

    print( 'max difference:' )
    print( '  ', torch.max( torch.abs( c_triton - c_torch  ) ) )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench( lambda: triton_bgemm(a, b ),
                                                  quantiles=quantiles )
    num_gflops =  2 * c.size(0) * c.size(1) * c.size(2) * a.size(2)
    num_gflops *= 1e-9

    print( "performance Triton: " )
    print( '  ms:     ', ms )
    print( '  min_ms: ', min_ms )
    print( '  max_ms: ', max_ms )
    print( '  GFLOPS: ', num_gflops / (ms*1e-3) )

    ms, min_ms, max_ms = triton.testing.do_bench( lambda: torch.matmul( a, b ),
                                                  quantiles=quantiles )

    print( "performance Torch: " )
    print( '  ms:     ', ms )
    print( '  min_ms: ', min_ms )
    print( '  max_ms: ', max_ms )
    print( '  GFLOPS: ', num_gflops / (ms*1e-3) )