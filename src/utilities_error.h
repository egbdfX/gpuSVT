/* utilities_error.h and utilities_mem.h are based on Error and Mem data structures that were created by Fred Dulwich and are used with his permission */


#ifndef UTILITIES_ERRORS_H_
#define UTILITIES_ERRORS_H_

/**
 * @file errors.h
 */

/**
 * @brief Processing function library error codes.
 */
enum Error
{
    //! No error.
    SUCCESS = 0,

    //! Generic runtime error.
    ERR_RUNTIME,

    //! Invalid function argument.
    ERR_INVALID_ARGUMENT,

    //! Unsupported data type.
    ERR_DATA_TYPE,

    //! Memory allocation failure.
    ERR_MEM_ALLOC_FAILURE,

    //! Memory copy failure.
    ERR_MEM_COPY_FAILURE,

    //! Unsupported memory location.
    ERR_MEM_LOCATION
};

typedef enum Error Error;

#endif
