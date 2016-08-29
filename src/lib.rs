#![allow(non_camel_case_types)]

extern crate libc;

use libc::{c_char, c_float, c_int, c_uint, c_void};
use libc::{size_t, uint64_t};

/// manually define unsigned int
pub type mx_uint = c_uint;
/// manually define float
pub type mx_float = c_float;

pub enum OpaqueNDArrayHandle {}
pub enum OpaqueFunctionHandle {}
pub enum OpaqueAtomicSymbolCreator {}
pub enum OpaqueSymbolHandle {}
pub enum OpaqueAtomicSymbolHandle {}
pub enum OpaqueExecutorHandle {}
pub enum OpaqueDataIterCreator {}
pub enum OpaqueDataIterHandle {}
pub enum OpaqueKVStoreHandle {}
pub enum OpaqueRecordIOHandle {}
pub enum OpaqueRtcHandle {}
pub enum OpaqueOptimizerCreator {}
pub enum OpaqueOptimizerHandle {}

/// handle to NDArray
pub type NDArrayHandle = *mut OpaqueNDArrayHandle;
/// handle to a mxnet narray function that changes NDArray
pub type FunctionHandle = *const OpaqueFunctionHandle;
/// handle to a function that takes param and creates symbol
pub type AtomicSymbolCreator = *mut OpaqueAtomicSymbolCreator;
/// handle to a symbol that can be bind as operator
pub type SymbolHandle = *mut OpaqueSymbolHandle;
/// handle to a AtomicSymbol
pub type AtomicSymbolHandle = *mut OpaqueAtomicSymbolHandle;
/// handle to an Executor
pub type ExecutorHandle = *mut OpaqueExecutorHandle;
/// handle a dataiter creator
pub type DataIterCreator = *mut OpaqueDataIterCreator;
/// handle to a DataIterator
pub type DataIterHandle = *mut OpaqueDataIterHandle;
/// handle to KVStore
pub type KVStoreHandle = *mut OpaqueKVStoreHandle;
/// handle to RecordIO
pub type RecordIOHandle = *mut OpaqueRecordIOHandle;
/// handle to MXRtc
pub type RtcHandle = *mut OpaqueRtcHandle;
/// handle to a function that takes param and creates optimizer
pub type OptimizerCreator = *mut OpaqueOptimizerCreator;
/// handle to Optimizer
pub type OptimizerHandle = *mut OpaqueOptimizerHandle;

pub type ExecutorMonitorCallback = extern "C" fn(*const c_char, NDArrayHandle, c_void);

#[repr(C)]
pub struct NativeOpInfo {
    pub forward: extern "C" fn(size: c_int,
                               ptrs: *mut *mut c_float,
                               ndims: *mut c_int,
                               shapes: *mut *mut c_uint,
                               tags: *mut c_int,
                               state: *mut c_void),
    pub backward: extern "C" fn(size: c_int,
                                ptrs: *mut *mut c_float,
                                ndims: *mut c_int,
                                shapes: *mut *mut c_uint,
                                tags: *mut c_int,
                                state: *mut c_void),
    pub infer_shape: extern "C" fn(size: c_int,
                                   ndims: *mut c_int,
                                   shapes: *mut *mut c_uint,
                                   state: *mut c_void),
    pub list_outputs: extern "C" fn(args: *mut *mut *mut c_char, state: *mut c_void),
    pub list_arguments: extern "C" fn(args: *mut *mut *mut c_char, state: *mut c_void),

    // all functions also pass a payload void* pointer
    pub p_forward: *mut c_void,
    pub p_backward: *mut c_void,
    pub p_infer_shape: *mut c_void,
    pub p_list_outputs: *mut c_void,
    pub p_list_arguments: *mut c_void,
}

#[repr(C)]
pub struct NDArrayOpInfo {
    pub forward: extern "C" fn(size: c_int,
                               ptrs: *mut *mut c_void,
                               tags: *mut c_int,
                               state: *mut c_void)
                               -> bool,
    pub backward: extern "C" fn(size: c_int,
                                ptrs: *mut *mut c_void,
                                tags: *mut c_int,
                                state: *mut c_void)
                                -> bool,
    pub infer_shape: extern "C" fn(num_input: c_int,
                                   ndims: *mut c_int,
                                   shapes: *mut *mut c_uint,
                                   state: *mut c_void)
                                   -> bool,
    pub list_outputs: extern "C" fn(outputs: *mut *mut *mut c_char, state: *mut c_void) -> bool,
    pub list_arguments: extern "C" fn(outputs: *mut *mut *mut c_char, state: *mut c_void) -> bool,
    pub declare_backward_dependency: extern "C" fn(out_grad: *const c_int,
                                                   in_data: *const c_int,
                                                   out_data: *const c_int,
                                                   num_deps: *mut c_int,
                                                   rdeps: *mut *mut c_int,
                                                   state: *mut c_void)
                                                   -> bool,

    // all functions also pass a payload void* pointer
    pub p_forward: *mut c_void,
    pub p_backward: *mut c_void,
    pub p_infer_shape: *mut c_void,
    pub p_list_outputs: *mut c_void,
    pub p_list_arguments: *mut c_void,
    pub p_declare_backward_dependency: *mut c_void,
}

#[repr(C)]
pub struct CustomOpInfo {
    pub forward: extern "C" fn(size: c_int,
                               ptrs: *mut *mut c_void,
                               tags: *mut c_int,
                               reqs: *const c_int,
                               is_train: bool,
                               state: *mut c_void)
                               -> bool,
    pub backward: extern "C" fn(size: c_int,
                                ptrs: *mut *mut c_void,
                                tags: *mut c_int,
                                reqs: *const c_int,
                                is_train: bool,
                                state: *mut c_void)
                                -> bool,
    pub del: extern "C" fn(*mut c_void /* state */) -> bool,

    // all functions also pass a payload void* pointer
    pub p_forward: *mut c_void,
    pub p_backward: *mut c_void,
    pub p_del: *mut c_void,
}

#[repr(C)]
pub struct CustomOpPropInfo {
    pub list_arguments: extern "C" fn(args: *mut *mut *mut c_char, state: *mut c_void) -> bool,
    pub list_outputs: extern "C" fn(outputs: *mut *mut *mut c_char, state: *mut c_void) -> bool,
    pub infer_shape: extern "C" fn(num_input: c_int,
                                   ndims: *mut c_int,
                                   shapes: *mut *mut c_uint,
                                   state: *mut c_void)
                                   -> bool,
    pub declare_backward_dependency: extern "C" fn(out_grad: *const c_int,
                                                   in_data: *const c_int,
                                                   out_data: *const c_int,
                                                   num_deps: *mut c_int,
                                                   rdeps: *mut *mut c_int,
                                                   state: *mut c_void)
                                                   -> bool,
    pub create_operator: extern "C" fn(*const c_char, // ctx
                                       c_int, // num_inputs
                                       *mut *mut c_uint, // shapes
                                       *mut c_int, // ndims
                                       *mut c_int, // dtypes
                                       *mut CustomOpInfo, // ret
                                       *mut c_void /* state */)
                                       -> bool,
    pub list_auxiliary_states: extern "C" fn(*mut *mut *mut c_char, // aux
                                             *mut c_void /* state */)
                                             -> bool,
    pub del: extern "C" fn(state: *mut c_void) -> bool,

    // all functions also pass a payload void* pointer
    pub p_list_arguments: *mut c_void,
    pub p_list_outputs: *mut c_void,
    pub p_infer_shape: *mut c_void,
    pub p_declare_backward_dependency: *mut c_void,
    pub p_create_operator: *mut c_void,
    pub p_list_auxiliary_states: *mut c_void,
    pub p_del: *mut c_void,
}

pub type CustomOpPropCreator = extern "C" fn(op_type: *const c_char,
                                             num_kwargs: c_int,
                                             keys: *const c_char,
                                             values: *const *const c_char,
                                             ret: *mut CustomOpPropInfo)
                                             -> bool;

/// user-defined updater for the kvstore
///
/// It's this updater's responsibility to delete recv and local
///
/// - param: the key
/// - param: recv the pushed value on this key
/// - param: local the value stored on local on this key
/// - param: handle The additional handle to the updater
pub type MXKVStoreUpdater = extern "C" fn(key: c_int,
                                          recv: NDArrayHandle,
                                          local: NDArrayHandle,
                                          handle: *mut c_void);

/// the prototype of a server controller
///
/// - param: head the head of the command
/// - param: body the body of the command
/// - param: controller_handle helper handle for implementing controller
pub type MXKVStoreServerController = extern "C" fn(head: c_int,
                                                   body: *const c_char,
                                                   controller_handle: *mut c_void);

#[link(name = "mxnet")]
extern "C" {

    /// return str message of the last error
    ///
    /// all function in this file will return 0 when success
    /// and -1 when an error occured,
    /// MXGetLastError can be called to retrieve the error
    ///
    /// this function is threadsafe and can be called by different thread
    ///
    /// -return: error info
    pub fn MXGetLastError() -> *const c_char;

    // -------------------------------------
    // Part 0: Global State setups
    // -------------------------------------

    /// Seed the global random number generators in mxnet.
    ///
    /// - param seed the random number seed.
    /// - return: 0 when success, -1 when failure happens.
    pub fn MXRandomSeed(seed: c_int) -> c_int;

    /// Notify the engine about a shutdown,
    ///
    /// This can help engine to print less messages into display.
    ///
    /// User do not have to call this function.
    ///
    /// - return: 0 when success, -1 when failure happens.
    pub fn MXNotifyShutdown() -> c_int;

    // -------------------------------------
    // Part 1: NDArray creation and deletion
    // -------------------------------------

    /// create a NDArray handle that is not initialized
    /// can be used to pass in as mutate variables
    /// to hold the result of NDArray
    ///
    /// - param: out the returning handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayCreateNone(out: *mut NDArrayHandle) -> c_int;

    /// create a NDArray with specified shape
    ///
    /// - param: shape the pointer to the shape
    /// - param: ndim the dimension of the shape
    /// - param: dev_type device type, specify device we want to take
    /// - param: dev_id the device id of the specific device
    /// - param: delay_alloc whether to delay allocation until
    ///   the narray is first mutated
    /// - param: out the returning handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayCreate(shape: *const mx_uint,
                           ndim: mx_uint,
                           dev_type: c_int,
                           dev_id: c_int,
                           delay_alloc: c_int,
                           out: *mut NDArrayHandle)
                           -> c_int;

    /// create a NDArray with specified shape and data type
    ///
    /// - param: shape the pointer to the shape
    /// - param: ndim the dimension of the shape
    /// - param: dev_type device type, specify device we want to take
    /// - param: dev_id the device id of the specific device
    /// - param: delay_alloc whether to delay allocation until
    ///   the narray is first mutated
    /// - param: dtype data type of created array
    /// - param: out the returning handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayCreateEx(shape: *const mx_uint,
                             ndim: mx_uint,
                             dev_type: c_int,
                             dev_id: c_int,
                             delay_alloc: c_int,
                             dtype: c_int,
                             out: *mut NDArrayHandle)
                             -> c_int;
    /// create a NDArray handle that is loaded from raw bytes.
    ///
    /// - param: buf the head of the raw bytes
    /// - param: size size of the raw bytes
    /// - param: out the returning handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayLoadFromRawBytes(buf: *const c_void,
                                     size: size_t,
                                     out: *mut NDArrayHandle)
                                     -> c_int;
    /// save the NDArray into raw bytes.
    ///
    /// - param: handle the NDArray handle
    /// - param: out_size size of the raw bytes
    /// - param: out_buf the head of returning memory bytes.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArraySaveRawBytes(handle: NDArrayHandle,
                                 out_size: *mut size_t,
                                 out_buf: *mut *const c_char)
                                 -> c_int;

    /// Save list of narray into the file.
    ///
    /// - param: fname name of the file.
    /// - param: num_args number of arguments to save.
    /// - param: args the array of NDArrayHandles to be saved.
    /// - param: keys the name of the NDArray, optional, can be NULL
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArraySave(fname: *const c_char,
                         num_args: mx_uint,
                         args: *mut NDArrayHandle,
                         keys: *const *const c_char)
                         -> c_int;

    /// Load list of narray from the file.
    ///
    /// - param: fname name of the file.
    /// - param: out_size number of narray loaded.
    /// - param: out_arr head of the returning narray handles.
    /// - param: out_name_size size of output name arrray.
    /// - param: out_names the names of returning NDArrays, can be NULL
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayLoad(fname: *const c_char,
                         out_size: *mut mx_uint,
                         out_arr: *mut *mut NDArrayHandle,
                         out_name_size: *mut mx_uint,
                         out_names: *mut *const *const c_char)
                         -> c_int;

    /// Perform a synchronize copy from a continugous CPU memory region.
    ///
    /// This function will call WaitToWrite before the copy is performed.
    /// This is useful to copy data from existing memory region that are
    /// not wrapped by NDArray(thus dependency not being tracked).
    ///
    /// - param: handle the NDArray handle
    /// - param: data the data source to copy from.
    /// - param: size the memory size we want to copy from.
    pub fn MXNDArraySyncCopyFromCPU(handle: NDArrayHandle,
                                    data: *const c_void,
                                    size: size_t)
                                    -> c_int;

    /// Perform a synchronize copyto a continugous CPU memory region.
    ///
    /// This function will call WaitToRead before the copy is performed.
    /// This is useful to copy data from existing memory region that are
    /// not wrapped by NDArray(thus dependency not being tracked).
    ///
    /// - param: handle the NDArray handle
    /// - param: data the data source to copy into.
    /// - param: size the memory size we want to copy into.
    pub fn MXNDArraySyncCopyToCPU(handle: NDArrayHandle, data: *mut c_void, size: size_t) -> c_int;

    /// Wait until all the pending writes with respect NDArray are finished.
    /// Always call this before read data out synchronizely.
    ///
    /// - param: handle the NDArray handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayWaitToRead(handle: NDArrayHandle) -> c_int;

    /// Wait until all the pending read/write with respect NDArray are finished.
    /// Always call this before write data into NDArray synchronizely.
    ///
    /// - param: handle the NDArray handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayWaitToWrite(handle: NDArrayHandle) -> c_int;

    /// wait until all delayed operations in
    /// the system is completed
    ///
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayWaitAll() -> c_int;

    /// free the narray handle
    ///
    /// - param: handle the handle to be freed
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayFree(handle: NDArrayHandle) -> c_int;

    /// Slice the NDArray along axis 0.
    ///
    /// - param: handle the handle to the narraya
    /// - param: slice_begin The beginning index of slice
    /// - param: slice_end The ending index of slice
    /// - param: out The NDArrayHandle of sliced NDArray
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArraySlice(handle: NDArrayHandle,
                          slice_begin: mx_uint,
                          slice_end: mx_uint,
                          out: *mut NDArrayHandle)
                          -> c_int;

    /// Index the NDArray along axis 0.
    ///
    /// - param: handle the handle to the narraya
    /// - param: idx the index
    /// - param: out The NDArrayHandle of sliced NDArray
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayAt(handle: NDArrayHandle, idx: mx_uint, out: *mut NDArrayHandle) -> c_int;

    /// Reshape the NDArray.
    ///
    /// - param: handle the handle to the narray
    /// - param: ndim number of dimensions of new shape
    /// - param: dims new shape
    /// - param: out the NDArrayHandle of reshaped NDArray
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayReshape(handle: NDArrayHandle,
                            ndim: c_int,
                            dims: *const c_int,
                            out: *mut NDArrayHandle)
                            -> c_int;

    /// get the shape of the array
    ///
    /// - param: handle the handle to the narray
    /// - param: out_dim the output dimension
    /// - param: out_pdata pointer holder to get data pointer of the shape
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayGetShape(handle: NDArrayHandle,
                             out_dim: *mut mx_uint,
                             out_pdata: *mut *const mx_uint)
                             -> c_int;

    /// get the content of the data in NDArray
    ///
    /// - param: handle the handle to the narray
    /// - param: out_pdata pointer holder to get pointer of data
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayGetData(handle: NDArrayHandle, out_pdata: *mut *mut mx_float) -> c_int;

    /// get the type of the data in NDArray
    ///
    /// - param: handle the handle to the narray
    /// - param: out_dtype pointer holder to get type of data
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayGetDType(handle: NDArrayHandle, out_dtype: *mut c_int) -> c_int;

    /// get the context of the NDArray
    ///
    /// - param: handle the handle to the narray
    /// - param: out_dev_type the output device type
    /// - param: out_dev_id the output device id
    /// - return: 0 when success, -1 when failure happens
    pub fn MXNDArrayGetContext(handle: NDArrayHandle,
                               out_dev_type: *mut c_int,
                               out_dev_id: *mut c_int)
                               -> c_int;

    // --------------------------------
    // Part 2: functions on NDArray
    // --------------------------------

    /// list all the available functions handles
    /// most user can use it to list all the needed functions
    ///
    /// - param: out_size the size of returned array
    /// - param: out_array the output function array
    /// - return: 0 when success, -1 when failure happens
    pub fn MXListFunctions(out_size: *mut mx_uint, out_array: *mut *mut FunctionHandle) -> c_int;

    /// get the function handle by name
    ///
    /// - param: name the name of the function
    /// - param: out the corresponding function handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXGetFunction(name: *const c_char, out: *mut FunctionHandle) -> c_int;

    /// Get the information of the function handle.
    ///
    /// - param: fun The function handle.
    /// - param: name The returned name of the function.
    /// - param: description The returned description of the function.
    /// - param: num_args Number of arguments.
    /// - param: arg_names Name of the arguments.
    /// - param: arg_type_infos Type informations about the arguments.
    /// - param: arg_descriptions Description information about the arguments.
    /// - param: return_type Return type of the function.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXFuncGetInfo(fun: FunctionHandle,
                         name: *mut *const c_char,
                         description: *mut *const c_char,
                         num_args: *mut mx_uint,
                         arg_names: *mut *const *const c_char,
                         arg_type_infos: *mut *const *const c_char,
                         arg_descriptions: *mut *const *const c_char,
                         return_type: *mut *const c_char)
                         -> c_int;

    /// get the argument requirements of the function
    ///
    /// - param: fun input function handle
    /// - param: num_use_vars how many NDArrays to be passed in as used_vars
    /// - param: num_scalars scalar variable is needed
    /// - param: num_mutate_vars how many NDArrays to be passed in as mutate_vars
    /// - param: type_mask the type mask of this function
    /// - return: 0 when success, -1 when failure happens
    /// - see: MXFuncInvoke
    pub fn MXFuncDescribe(fun: FunctionHandle,
                          num_use_vars: *mut mx_uint,
                          num_scalars: *mut mx_uint,
                          num_mutate_vars: *mut mx_uint,
                          type_mask: *mut c_int)
                          -> c_int;

    /// invoke a function, the array size of passed in arguments
    /// must match the values in the
    ///
    /// - param: fun the function
    /// - param: use_vars the normal arguments passed to function
    /// - param: scalar_args the scalar qarguments
    /// - param: mutate_vars the mutate arguments
    /// - return: 0 when success, -1 when failure happens
    /// - see: MXFuncDescribeArgs
    pub fn MXFuncInvoke(fun: FunctionHandle,
                        use_vars: *mut NDArrayHandle,
                        scalar_args: *mut mx_float,
                        mutate_vars: *mut NDArrayHandle)
                        -> c_int;

    /// invoke a function, the array size of passed in arguments
    /// must match the values in the
    ///
    /// - param: fun the function
    /// - param: use_vars the normal arguments passed to function
    /// - param: scalar_args the scalar qarguments
    /// - param: mutate_vars the mutate arguments
    /// - param: num_params number of keyword parameters
    /// - param: param_keys keys for keyword parameters
    /// - param: param_vals values for keyword parameters
    /// - return: 0 when success, -1 when failure happens
    /// - see: MXFuncDescribeArgs
    pub fn MXFuncInvokeEx(fun: FunctionHandle,
                          use_vars: *mut NDArrayHandle,
                          scalar_args: *mut mx_float,
                          mutate_vars: *mut NDArrayHandle,
                          num_params: c_int,
                          param_keys: *mut *mut c_char,
                          param_vals: *mut *mut c_char)
                          -> c_int;

    // --------------------------------------------
    // Part 3: symbolic configuration generation
    // --------------------------------------------

    /// list all the available AtomicSymbolEntry
    ///
    /// - param: out_size the size of returned array
    /// - param: out_array the output AtomicSymbolCreator array
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolListAtomicSymbolCreators(out_size: *mut mx_uint,
                                            out_array: *mut *mut AtomicSymbolCreator)
                                            -> c_int;

    /// Get the name of an atomic symbol.
    ///
    /// - param: creator the AtomicSymbolCreator.
    /// - param: name The returned name of the creator.
    pub fn MXSymbolGetAtomicSymbolName(creator: AtomicSymbolCreator,
                                       name: *mut *const c_char)
                                       -> c_int;

    /// Get the detailed information about atomic symbol.
    ///
    /// - param: creator the AtomicSymbolCreator.
    /// - param: name The returned name of the creator.
    /// - param: description The returned description of the symbol.
    /// - param: num_args Number of arguments.
    /// - param: arg_names Name of the arguments.
    /// - param: arg_type_infos Type informations about the arguments.
    /// - param: arg_descriptions Description information about the arguments.
    /// - param: key_var_num_args The keyword argument for specifying variable number of arguments.
    ///          When this parameter has non-zero length, the function allows variable number
    ///          of positional arguments, and will need the caller to pass it in in
    ///          MXSymbolCreateAtomicSymbol,
    ///          With key = key_var_num_args, and value = number of positional arguments.
    /// - param: return_type Return type of the function, can be Symbol or Symbol[]
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolGetAtomicSymbolInfo(creator: AtomicSymbolCreator,
                                       name: *mut *const c_char,
                                       description: *mut *const c_char,
                                       num_args: *mut mx_uint,
                                       arg_names: *mut *const *const c_char,
                                       arg_type_infos: *mut *const *const c_char,
                                       arg_descriptions: *mut *const *const c_char,
                                       key_var_num_args: *mut *const c_char,
                                       return_type: *mut *const c_char)
                                       -> c_int;

    /// Create an AtomicSymbol.
    ///
    /// - param: creator the AtomicSymbolCreator
    /// - param: num_param the number of parameters
    /// - param: keys the keys to the params
    /// - param: vals the vals of the params
    /// - param: out pointer to the created symbol handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolCreateAtomicSymbol(creator: AtomicSymbolCreator,
                                      num_param: mx_uint,
                                      keys: *const *const c_char,
                                      vals: *const *const c_char,
                                      out: *mut SymbolHandle)
                                      -> c_int;

    /// Create a Variable Symbol.
    ///
    /// - param: name name of the variable
    /// - param: out pointer to the created symbol handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolCreateVariable(name: *const c_char, out: *mut SymbolHandle) -> c_int;

    /// Create a Symbol by grouping list of symbols together
    ///
    /// - param: num_symbols number of symbols to be grouped
    /// - param: symbols array of symbol handles
    /// - param: out pointer to the created symbol handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolCreateGroup(num_symbols: mx_uint,
                               symbols: *mut SymbolHandle,
                               out: *mut SymbolHandle)
                               -> c_int;

    /// Load a symbol from a json file.
    ///
    /// - param: fname the file name.
    /// - param: out the output symbol.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolCreateFromFile(fname: *const c_char, out: *mut SymbolHandle) -> c_int;

    /// Load a symbol from a json string.
    ///
    /// - param: json the json string.
    /// - param: out the output symbol.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolCreateFromJSON(json: *const c_char, out: *mut SymbolHandle) -> c_int;

    /// Save a symbol into a json file.
    ///
    /// - param: symbol the input symbol.
    /// - param: fname the file name.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolSaveToFile(symbol: SymbolHandle, fname: *const c_char) -> c_int;

    /// Save a symbol into a json string
    ///
    /// - param: symbol the input symbol.
    /// - param: out_json output json string.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolSaveToJSON(symbol: SymbolHandle, out_json: *mut *const c_char) -> c_int;

    /// Free the symbol handle.
    ///
    /// - param: symbol the symbol
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolFree(symbol: SymbolHandle) -> c_int;

    /// Copy the symbol to another handle
    ///
    /// - param: symbol the source symbol
    /// - param: out used to hold the result of copy
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolCopy(symbol: SymbolHandle, out: *mut SymbolHandle) -> c_int;

    /// Print the content of symbol, used for debug.
    ///
    /// - param: symbol the symbol
    /// - param: out_str pointer to hold the output string of the printing.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolPrint(symbol: SymbolHandle, out_str: *mut *const c_char) -> c_int;

    /// Get string name from symbol
    ///
    /// - param: symbol the source symbol
    /// - param: out The result name.
    /// - param: success Whether the result is contained in out.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolGetName(symbol: SymbolHandle,
                           out: *mut *const c_char,
                           success: *mut c_int)
                           -> c_int;

    /// Get string attribute from symbol
    ///
    /// - param: symbol the source symbol
    /// - param: key The key of the symbol.
    /// - param: out The result attribute, can be NULL if the attribute do not exist.
    /// - param: success Whether the result is contained in out.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolGetAttr(symbol: SymbolHandle,
                           key: *const c_char,
                           out: *mut *const c_char,
                           success: *mut c_int)
                           -> c_int;

    /// Set string attribute from symbol.
    ///
    /// NOTE: Setting attribute to a symbol can affect the semantics
    /// (mutable/immutable) of symbolic graph.
    ///
    /// Safe recommendaton: use immutable graph
    /// - Only allow set attributes during creation of new symbol as optional parameter
    ///
    /// Mutable graph (be careful about the semantics):
    /// - Allow set attr at any point.
    /// - Mutating an attribute of some common node of two graphs can cause confusion from user.
    ///
    /// - param: symbol the source symbol
    /// - param: key The key of the symbol.
    /// - param: value The value to be saved.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolSetAttr(symbol: SymbolHandle, key: *const c_int, value: *const c_int) -> c_int;

    /// Get all attributes from symbol, including all descendents.
    ///
    /// - param: symbol the source symbol
    /// - param: out_size The number of output attributes
    /// - param: out 2*out_size strings representing key value pairs.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolListAttr(symbol: SymbolHandle,
                            out_size: *mut mx_uint,
                            out: *mut *const *const c_char)
                            -> c_int;

    /// Get all attributes from symbol, excluding descendents.
    ///
    /// - param: symbol the source symbol
    /// - param: out_size The number of output attributes
    /// - param: out 2*out_size strings representing key value pairs.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolListAttrShallow(symbol: SymbolHandle,
                                   out_size: *mut mx_uint,
                                   out: *mut *const *const c_char)
                                   -> c_int;

    /// List arguments in the symbol.
    ///
    /// - param: symbol the symbol
    /// - param: out_size output size
    /// - param: out_str_array pointer to hold the output string array
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolListArguments(symbol: SymbolHandle,
                                 out_size: *mut mx_uint,
                                 out_str_array: *mut *const *const c_char)
                                 -> c_int;

    /// List returns in the symbol.
    ///
    /// - param: symbol the symbol
    /// - param: out_size output size
    /// - param: out_str_array pointer to hold the output string array
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolListOutputs(symbol: SymbolHandle,
                               out_size: *mut mx_uint,
                               out_str_array: *mut *const *const c_char)
                               -> c_int;

    /// Get a symbol that contains all the internals.
    ///
    /// - param: symbol The symbol
    /// - param: out The output symbol whose outputs are all the internals.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolGetInternals(symbol: SymbolHandle, out: *mut SymbolHandle) -> c_int;

    /// Get index-th outputs of the symbol.
    ///
    /// - param: symbol The symbol
    /// - param: index the Index of the output.
    /// - param: out The output symbol whose outputs are the index-th symbol.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolGetOutput(symbol: SymbolHandle,
                             index: mx_uint,
                             out: *mut SymbolHandle)
                             -> c_int;

    /// List auxiliary states in the symbol.
    ///
    /// - param: symbol the symbol
    /// - param: out_size output size
    /// - param: out_str_array pointer to hold the output string array
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolListAuxiliaryStates(symbol: SymbolHandle,
                                       out_size: *mut mx_uint,
                                       out_str_array: *mut *const *const c_char)
                                       -> c_int;

    /// Compose the symbol on other symbols.
    ///
    /// This function will change the sym hanlde.
    /// To achieve function apply behavior, copy the symbol first
    /// before apply.
    ///
    /// - param: sym the symbol to apply
    /// - param: name the name of symbol
    /// - param: num_args number of arguments
    /// - param: keys the key of keyword args (optional)
    /// - param: args arguments to sym
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolCompose(sym: SymbolHandle,
                           name: *const c_char,
                           num_args: mx_uint,
                           keys: *const *const c_char,
                           args: *const SymbolHandle)
                           -> c_int;

    /// Get the gradient graph of the symbol
    ///
    /// - param: sym the symbol to get gradient
    /// - param: num_wrt number of arguments to get gradient
    /// - param: wrt the name of the arguments to get gradient
    /// - param: out the returned symbol that has gradient
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolGrad(sym: SymbolHandle,
                        num_wrt: mx_uint,
                        wrt: *const *const c_char,
                        out: *const SymbolHandle)
                        -> c_int;

    /// infer shape of unknown input shapes given the known one.
    ///
    /// The shapes are packed into a CSR matrix represented by arg_ind_ptr
    /// and arg_shape_data
    ///
    /// The call will be treated as a kwargs call if key != nullptr or
    /// num_args==0, otherwise it is positional.
    ///
    /// - param: sym symbol handle
    /// - param: num_args numbe of input arguments.
    /// - param: keys the key of keyword args (optional)
    /// - param: arg_ind_ptr the head pointer of the rows in CSR
    /// - param: arg_shape_data the content of the CSR
    /// - param: in_shape_size sizeof the returning array of in_shapes
    /// - param: in_shape_ndim returning array of shape dimensions of eachs input shape.
    /// - param: in_shape_data returning array of pointers to head of the input shape.
    /// - param: out_shape_size sizeof the returning array of out_shapes
    /// - param: out_shape_ndim returning array of shape dimensions of eachs input shape.
    /// - param: out_shape_data returning array of pointers to head of the input shape.
    /// - param: aux_shape_size sizeof the returning array of aux_shapes
    /// - param: aux_shape_ndim returning array of shape dimensions of eachs auxiliary shape.
    /// - param: aux_shape_data returning array of pointers to head of the auxiliary shape.
    /// - param: complete whether infer shape completes or more information is needed.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolInferShape(sym: SymbolHandle,
                              num_args: mx_uint,
                              keys: *const *const c_char,
                              arg_ind_ptr: *const mx_uint,
                              arg_shape_data: *const mx_uint,
                              in_shape_size: *mut mx_uint,
                              in_shape_ndim: *const *const mx_uint,
                              in_shape_data: *const *const *const mx_uint,
                              out_shape_size: *mut mx_uint,
                              out_shape_ndim: *const *const mx_uint,
                              out_shape_data: *const *const *const mx_uint,
                              aux_shape_size: *mut mx_uint,
                              aux_shape_ndim: *const *const mx_uint,
                              aux_shape_data: *const *const *const mx_uint,
                              complete: *mut c_int);

    /// partially infer shape of unknown input shapes given the known one.
    ///
    /// Return partially inferred results if not all shapes could be inferred.
    ///
    /// The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
    ///
    /// The call will be treated as a kwargs call if key != nullptr or num_args==0,
    /// otherwise it is positional.
    ///
    /// - param: sym symbol handle
    /// - param: num_args numbe of input arguments.
    /// - param: keys the key of keyword args (optional)
    /// - param: arg_ind_ptr the head pointer of the rows in CSR
    /// - param: arg_shape_data the content of the CSR
    /// - param: in_shape_size sizeof the returning array of in_shapes
    /// - param: in_shape_ndim returning array of shape dimensions of eachs input shape.
    /// - param: in_shape_data returning array of pointers to head of the input shape.
    /// - param: out_shape_size sizeof the returning array of out_shapes
    /// - param: out_shape_ndim returning array of shape dimensions of eachs input shape.
    /// - param: out_shape_data returning array of pointers to head of the input shape.
    /// - param: aux_shape_size sizeof the returning array of aux_shapes
    /// - param: aux_shape_ndim returning array of shape dimensions of eachs auxiliary shape.
    /// - param: aux_shape_data returning array of pointers to head of the auxiliary shape.
    /// - param: complete whether infer shape completes or more information is needed.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolInferShapePartial(sym: SymbolHandle,
                                     num_args: mx_uint,
                                     keys: *const *const c_int,
                                     arg_ind_ptr: *const mx_uint,
                                     arg_shape_data: *const mx_uint,
                                     in_shape_size: *mut mx_uint,
                                     tin_shape_ndim: *const *const mx_uint,
                                     ntin_shape_data: *const *const *const mx_uint,
                                     out_shape_size: *mut mx_uint,
                                     tout_shape_ndim: *const *const mx_uint,
                                     ntout_shape_data: *const *const *const mx_uint,
                                     aux_shape_size: *mut mx_uint,
                                     taux_shape_ndim: *const *const mx_uint,
                                     ntaux_shape_data: *const *const *const mx_uint,
                                     complete: *mut c_int);

    /// infer type of unknown input types given the known one.
    ///
    /// The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
    ///
    /// The call will be treated as a kwargs call if key != nullptr or num_args==0,
    /// otherwise it is positional.
    ///
    /// - param: sym symbol handle
    /// - param: num_args numbe of input arguments.
    /// - param: keys the key of keyword args (optional)
    /// - param: arg_type_data the content of the CSR
    /// - param: in_type_size sizeof the returning array of in_types
    /// - param: in_type_data returning array of pointers to head of the input type.
    /// - param: out_type_size sizeof the returning array of out_types
    /// - param: out_type_data returning array of pointers to head of the input type.
    /// - param: aux_type_size sizeof the returning array of aux_types
    /// - param: aux_type_data returning array of pointers to head of the auxiliary type.
    /// - param: complete whether infer type completes or more information is needed.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXSymbolInferType(sym: SymbolHandle,
                             num_args: mx_uint,
                             keys: *const *const c_char,
                             arg_type_data: *const c_int,
                             in_type_size: *mut mx_uint,
                             in_type_data: *const *const c_int,
                             out_type_size: *mut mx_uint,
                             out_type_data: *const *const c_int,
                             aux_type_size: *mut mx_uint,
                             aux_type_data: *const *const c_int,
                             complete: *mut c_int)
                             -> c_int;

    // --------------------------------------------
    // Part 4: Executor interface
    // --------------------------------------------

    /// Delete the executor
    ///
    /// - param: handle the executor.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorFree(handle: ExecutorHandle) -> c_int;

    /// Print the content of execution plan, used for debug.
    ///
    /// - param: handle the executor.
    /// - param: out_str pointer to hold the output string of the printing.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorPrint(handle: ExecutorHandle, out_str: *mut *const c_char) -> c_int;

    /// Executor forward method
    ///
    /// - param: handle executor handle
    /// - param: is_train bool value to indicate whether the forward pass is for evaluation
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorForward(handle: ExecutorHandle, is_train: c_int) -> c_int;

    /// Excecutor run backward
    ///
    /// - param: handle execute handle
    /// - param: len lenth
    /// - param: head_grads NDArray handle for heads' gradient
    ///
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorBackward(handle: ExecutorHandle,
                              len: mx_uint,
                              head_grads: *mut NDArrayHandle)
                              -> c_int;

    /// Get executor's head NDArray
    ///
    /// - param: handle executor handle
    /// - param: out_size output narray vector size
    /// - param: out out put narray handles
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorOutputs(handle: ExecutorHandle,
                             out_size: *mut mx_uint,
                             out: *mut *mut NDArrayHandle)
                             -> c_int;

    /// Generate Executor from symbol
    ///
    /// - param: symbol_handle symbol handle
    /// - param: dev_type device type
    /// - param: dev_id device id
    /// - param: len length
    /// - param: in_args in args array
    /// - param: arg_grad_store arg grads handle array
    /// - param: grad_req_type grad req array
    /// - param: aux_states_len length of auxiliary states
    /// - param: aux_states auxiliary states array
    /// - param: out output executor handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorBind(symbol_handle: SymbolHandle,
                          dev_type: c_int,
                          dev_id: c_int,
                          len: mx_uint,
                          in_args: *mut NDArrayHandle,
                          arg_grad_store: *mut NDArrayHandle,
                          grad_req_type: *mut mx_uint,
                          aux_states_len: mx_uint,
                          aux_states: *mut NDArrayHandle,
                          out: *mut ExecutorHandle)
                          -> c_int;

    /// Generate Executor from symbol,
    ///
    /// This is advanced function, allow specify group2ctx map.
    /// The user can annotate "ctx_group" attribute to name each group.
    ///
    /// - param: symbol_handle symbol handle
    /// - param: dev_type device type of default context
    /// - param: dev_id device id of default context
    /// - param: num_map_keys size of group2ctx map
    /// - param: map_keys keys of group2ctx map
    /// - param: map_dev_types device type of group2ctx map
    /// - param: map_dev_ids device id of group2ctx map
    /// - param: len length
    /// - param: in_args in args array
    /// - param: arg_grad_store arg grads handle array
    /// - param: grad_req_type grad req array
    /// - param: aux_states_len length of auxiliary states
    /// - param: aux_states auxiliary states array
    /// - param: out output executor handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorBindX(symbol_handle: SymbolHandle,
                           dev_type: c_int,
                           dev_id: c_int,
                           num_map_keys: mx_uint,
                           map_keys: *const *const c_char,
                           map_dev_types: *const c_int,
                           map_dev_ids: *const c_int,
                           len: mx_uint,
                           in_args: *mut NDArrayHandle,
                           arg_grad_store: *mut NDArrayHandle,
                           grad_req_type: *mut mx_uint,
                           aux_states_len: mx_uint,
                           aux_states: *mut NDArrayHandle,
                           out: *mut ExecutorHandle)
                           -> c_int;

    /// Generate Executor from symbol,
    ///
    /// This is advanced function, allow specify group2ctx map.
    /// The user can annotate "ctx_group" attribute to name each group.
    ///
    /// - param: symbol_handle symbol handle
    /// - param: dev_type device type of default context
    /// - param: dev_id device id of default context
    /// - param: num_map_keys size of group2ctx map
    /// - param: map_keys keys of group2ctx map
    /// - param: map_dev_types device type of group2ctx map
    /// - param: map_dev_ids device id of group2ctx map
    /// - param: len length
    /// - param: in_args in args array
    /// - param: arg_grad_store arg grads handle array
    /// - param: grad_req_type grad req array
    /// - param: aux_states_len length of auxiliary states
    /// - param: aux_states auxiliary states array
    /// - param: shared_exec input executor handle for memory sharing
    /// - param: out output executor handle
    /// - return: 0 when success, -1 when failure happens
    pub fn MXExecutorBindEX(symbol_handle: SymbolHandle,
                            dev_type: c_int,
                            dev_id: c_int,
                            num_map_keys: mx_uint,
                            map_keys: *const *const c_char,
                            map_dev_types: *const c_int,
                            map_dev_ids: *const c_int,
                            len: mx_uint,
                            in_args: *mut NDArrayHandle,
                            arg_grad_store: *mut NDArrayHandle,
                            grad_req_type: *mut mx_uint,
                            aux_states_len: mx_uint,
                            aux_states: *mut NDArrayHandle,
                            shared_exec: ExecutorHandle,
                            out: *mut ExecutorHandle)
                            -> c_int;

    /// set a call back to notify the completion of operation
    pub fn MXExecutorSetMonitorCallback(handle: ExecutorHandle,
                                        callback: ExecutorMonitorCallback,
                                        callback_handle: *mut c_void)
                                        -> c_int;

    // --------------------------------------------
    // Part 5: IO Interface
    // --------------------------------------------

    /// List all the available iterator entries
    ///
    /// - param: out_size the size of returned iterators
    /// - param: out_array the output iteratos entries
    /// - return: 0 when success, -1 when failure happens
    pub fn MXListDataIters(out_size: *mut mx_uint, out_array: *mut *mut DataIterCreator) -> c_int;

    /// Init an iterator, init with parameters
    /// the array size of passed in arguments
    ///
    /// - param: handle of the iterator creator
    /// - param: num_param number of parameter
    /// - param: keys parameter keys
    /// - param: vals parameter values
    /// - param: out resulting iterator
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterCreateIter(handle: DataIterCreator,
                                num_param: mx_uint,
                                keys: *const *const c_char,
                                vals: *const *const c_char,
                                out: *mut DataIterHandle)
                                -> c_int;

    /// Get the detailed information about data iterator.
    ///
    /// - param: creator the DataIterCreator.
    /// - param: name The returned name of the creator.
    /// - param: description The returned description of the symbol.
    /// - param: num_args Number of arguments.
    /// - param: arg_names Name of the arguments.
    /// - param: arg_type_infos Type informations about the arguments.
    /// - param: arg_descriptions Description information about the arguments.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterGetIterInfo(creator: DataIterCreator,
                                 name: *mut *const c_char,
                                 description: *mut *const c_char,
                                 num_args: *mut mx_uint,
                                 arg_names: *mut *const *const c_char,
                                 arg_type_infos: *mut *const *const c_char,
                                 arg_descriptions: *mut *const *const c_char)
                                 -> c_int;

    /// Free the handle to the IO module
    ///
    /// - param: handle the handle pointer to the data iterator
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterFree(handle: DataIterHandle) -> c_int;

    /// Move iterator to next position
    ///
    /// - param: handle the handle to iterator
    /// - param: out return value of next
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterNext(handle: DataIterHandle, out: *mut c_int) -> c_int;

    /// Call iterator.Reset
    ///
    /// - param: handle the handle to iterator
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterBeforeFirst(handle: DataIterHandle) -> c_int;

    /// Get the handle to the NDArray of underlying data
    ///
    /// - param: handle the handle pointer to the data iterator
    /// - param: out handle to underlying data NDArray
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterGetData(handle: DataIterHandle, out: *mut NDArrayHandle) -> c_int;

    /// Get the image index by array.
    ///
    /// - param: handle the handle pointer to the data iterator
    /// - param: out_index output index of the array.
    /// - param: out_size output size of the array.
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterGetIndex(handle: DataIterHandle,
                              out_index: *mut *mut uint64_t,
                              out_size: *mut uint64_t)
                              -> c_int;

    /// Get the padding number in current data batch
    ///
    /// - param: handle the handle pointer to the data iterator
    /// - param: pad pad number ptr
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterGetPadNum(handle: DataIterHandle, pad: *const c_int) -> c_int;

    /// Get the handle to the NDArray of underlying label
    ///
    /// - param: handle the handle pointer to the data iterator
    /// - param: out the handle to underlying label NDArray
    /// - return: 0 when success, -1 when failure happens
    pub fn MXDataIterGetLabel(handle: DataIterHandle, out: *mut NDArrayHandle) -> c_int;

    // --------------------------------------------
    // Part 6: basic KVStore interface
    // --------------------------------------------

    /// Initialized ps-lite environment variables
    ///
    /// - param: num_vars number of variables to initialize
    /// - param: keys environment keys
    /// - param: vals environment values
    pub fn MXInitPSEnv(num_vars: mx_uint,
                       keys: *mut *const c_char,
                       vals: *mut *const c_char)
                       -> c_int;


    /// Create a kvstore
    ///
    /// - param: type the type of KVStore
    /// - param: out The output type of KVStore
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreCreate(kvtype: *const c_char, out: *mut KVStoreHandle) -> c_int;

    /// Delete a KVStore handle.
    ///
    /// - param: handle handle to the kvstore
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreFree(handle: KVStoreHandle) -> c_int;

    /// Init a list of (key,value) pairs in kvstore
    ///
    /// - param: handle handle to the kvstore
    /// - param: num the number of key-value pairs
    /// - param: keys the list of keys
    /// - param: vals the list of values
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreInit(handle: KVStoreHandle,
                         num: mx_uint,
                         keys: *const c_int,
                         vals: *mut NDArrayHandle)
                         -> c_int;

    /// Push a list of (key,value) pairs to kvstore
    ///
    /// - param: handle handle to the kvstore
    /// - param: num the number of key-value pairs
    /// - param: keys the list of keys
    /// - param: vals the list of values
    /// - param: priority the priority of the action
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStorePush(handle: KVStoreHandle,
                         num: mx_uint,
                         keys: *const c_int,
                         vals: *mut NDArrayHandle,
                         priority: c_int)
                         -> c_int;

    /// pull a list of (key, value) pairs from the kvstore
    ///
    /// - param: handle handle to the kvstore
    /// - param: num the number of key-value pairs
    /// - param: keys the list of keys
    /// - param: vals the list of values
    /// - param: priority the priority of the action
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStorePull(handle: KVStoreHandle,
                         num: mx_uint,
                         keys: c_int,
                         vals: *mut NDArrayHandle,
                         priority: c_int)
                         -> c_int;

    /// register an push updater
    ///
    /// - param: handle handle to the KVStore
    /// - param: updater udpater function
    /// - param: updater_handle The additional handle used to invoke the updater
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreSetUpdater(handle: KVStoreHandle,
                               updater: MXKVStoreUpdater,
                               updater_handle: *mut c_void)
                               -> c_int;

    /// get the type of the kvstore
    ///
    /// - param: handle handle to the KVStore
    /// - param: type a string type
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreGetType(handle: KVStoreHandle, kvtype: *mut *const c_char) -> c_int;

    // --------------------------------------------
    // Part 6: advanced KVStore for multi-machines
    // --------------------------------------------

    /// return The rank of this node in its group, which is in [0, GroupSize).
    ///
    /// - param: handle handle to the KVStore
    /// - param: ret the node rank
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreGetRank(handle: KVStoreHandle, ret: *mut c_int) -> c_int;

    /// return The number of nodes in this group, which is
    ///
    /// - number of workers if if `IsWorkerNode() == true`,
    /// - number of servers if if `IsServerNode() == true`,
    /// - 1 if `IsSchedulerNode() == true`,
    /// - param: handle handle to the KVStore
    /// - param: ret the group size
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreGetGroupSize(handle: KVStoreHandle, ret: *mut c_int) -> c_int;

    /// return whether or not this process is a worker node.
    ///
    /// - param: ret 1 for yes, 0 for no
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreIsWorkerNode(ret: *mut c_int) -> c_int;


    /// return whether or not this process is a server node.
    ///
    /// - param: ret 1 for yes, 0 for no
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreIsServerNode(ret: *mut c_int) -> c_int;


    /// return whether or not this process is a scheduler node.
    ///
    /// - param: ret 1 for yes, 0 for no
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreIsSchedulerNode(ret: *mut c_int) -> c_int;

    /// global barrier among all worker machines
    ///
    /// - param: handle handle to the KVStore
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreBarrier(handle: KVStoreHandle) -> c_int;

    /// whether to do barrier when finalize
    ///
    /// - param: handle handle to the KVStore
    /// - param: barrier_before_exit whether to do barrier when kvstore finalize
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreSetBarrierBeforeExit(handle: KVStoreHandle,
                                         barrier_before_exit: c_int)
                                         -> c_int;

    /// Run as server (or scheduler)
    ///
    /// - param: handle handle to the KVStore
    /// - param: controller the user-defined server controller
    /// - param: controller_handle helper handle for implementing controller
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreRunServer(handle: KVStoreHandle,
                              controller: MXKVStoreServerController,
                              controller_handle: *mut c_void)
                              -> c_int;

    /// Send a command to all server nodes
    ///
    /// - param: handle handle to the KVStore
    /// - param: cmd_id the head of the command
    /// - param: cmd_body the body of the command
    /// - return: 0 when success, -1 when failure happens
    pub fn MXKVStoreSendCommmandToServers(handle: KVStoreHandle,
                                          cmd_id: c_int,
                                          cmd_body: *const c_char)
                                          -> c_int;

    /// Get the number of ps dead node(s) specified by {node_id}
    ///
    /// - param: handle handle to the KVStore
    /// - param: node_id Can be a node group or a single node.
    ///          kScheduler = 1, kServerGroup = 2, kWorkerGroup = 4
    /// - param: number Ouptut number of dead nodes
    /// - param: timeout_sec A node fails to send heartbeart in {timeout_sec} seconds
    ///          will be presumed as 'dead'. Default is 60 seconds.
    pub fn MXKVStoreGetNumDeadNode(handle: KVStoreHandle,
                                   node_id: c_int,
                                   number: *mut c_int,
                                   timeout_sec: c_int)
                                   -> c_int;

    /// Create a RecordIO writer object
    ///
    /// - param: uri path to file
    /// - param: out handle pointer to the created object
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOWriterCreate(uri: *const c_char, out: *mut RecordIOHandle) -> c_int;

    /// Delete a RecordIO writer object
    ///
    /// - param: handle handle to RecordIO object
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOWriterFree(handle: RecordIOHandle) -> c_int;

    /// Write a record to a RecordIO object
    ///
    /// - param: handle handle to RecordIO object
    /// - param: buf buffer to write
    /// - param: size size of buffer
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOWriterWriteRecord(handle: *mut RecordIOHandle,
                                       buf: *const c_char,
                                       size: size_t)
                                       -> c_int;

    /// Get the current writer pointer position
    ///
    /// - param: handle handle to RecordIO object
    /// - param: pos handle to output position
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOWriterTell(handle: *mut RecordIOHandle, pos: *mut size_t) -> c_int;

    /// Create a RecordIO reader object
    ///
    /// - param: uri path to file
    /// - param: out handle pointer to the created object
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOReaderCreate(uri: *const c_char, out: *mut RecordIOHandle) -> c_int;

    /// Delete a RecordIO reader object
    ///
    /// - param: handle handle to RecordIO object
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOReaderFree(handle: *mut RecordIOHandle) -> c_int;

    /// Write a record to a RecordIO object
    ///
    /// - param: handle handle to RecordIO object
    /// - param: buf pointer to return buffer
    /// - param: size point to size of buffer
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOReaderReadRecord(handle: *mut RecordIOHandle,
                                      buf: *mut *const c_char,
                                      size: *mut size_t)
                                      -> c_int;

    /// Set the current reader pointer position
    ///
    /// - param: handle handle to RecordIO object
    /// - param: pos target position
    /// - return: 0 when success, -1 when failure happens
    pub fn MXRecordIOReaderSeek(handle: *mut RecordIOHandle, pos: size_t) -> c_int;

    /// Create a MXRtc object
    pub fn MXRtcCreate(name: *mut c_char,
                       num_input: mx_uint,
                       num_output: mx_uint,
                       input_names: *mut *mut c_char,
                       output_names: *mut *mut c_char,
                       inputs: *mut NDArrayHandle,
                       outputs: *mut NDArrayHandle,
                       kernel: *mut c_char,
                       out: *mut RtcHandle)
                       -> c_int;

    /// Run cuda kernel
    pub fn MXRtcPush(handle: RtcHandle,
                     num_input: mx_uint,
                     num_output: mx_uint,
                     inputs: *mut NDArrayHandle,
                     outputs: *mut NDArrayHandle,
                     gridDimX: mx_uint,
                     gridDimY: mx_uint,
                     gridDimZ: mx_uint,
                     blockDimX: mx_uint,
                     blockDimY: mx_uint,
                     blockDimZ: mx_uint)
                     -> c_int;

    /// Delete a MXRtc object
    pub fn MXRtcFree(handle: RtcHandle) -> c_int;

    pub fn MXOptimizerFindCreator(key: *const c_char, out: *mut OptimizerCreator) -> c_int;

    pub fn MXOptimizerCreateOptimizer(creator: OptimizerCreator,
                                      num_param: mx_uint,
                                      keys: *const *const c_char,
                                      vals: *const *const c_char,
                                      out: *mut OptimizerHandle)
                                      -> c_int;

    pub fn MXOptimizerFree(handle: OptimizerHandle) -> c_int;

    pub fn MXOptimizerUpdate(handle: OptimizerHandle,
                             index: c_int,
                             weight: NDArrayHandle,
                             grad: NDArrayHandle,
                             lr: mx_float,
                             wd: mx_float)
                             -> c_int;

    pub fn MXCustomOpRegister(op_type: *const c_char, creator: CustomOpPropCreator) -> c_int;

}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
