/**
 * @template {(...args: any[]) => any} T
 * @param {T} callback 
 * @param {number} waitInMs 
 * @returns {(...args: Parameters<T>) => void}
 */
const debounce = (callback, waitInMs) => {
    let timeoutId = null;

    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        callback(...args);
      }, waitInMs);
    };
  }

export default debounce