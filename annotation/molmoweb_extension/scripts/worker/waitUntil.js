
/**
 * 
 * @param {(...args: any[]) => boolean} conditionFunction 
 * @param {number | undefined} maxRetries 
 * @returns {Promise<void>}
 */
function waitUntil(conditionFunction, maxRetries = 3) {
    let retries = 0;

    const poll = (resolve) => {
        if (conditionFunction() || retries >= maxRetries) {
            resolve();
        } else {
            retries++;
            setTimeout(() => poll(resolve), 100);
        }
    }
    
    return new Promise(poll);
}

export default waitUntil;