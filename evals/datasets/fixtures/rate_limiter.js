// Token-bucket rate limiter used by the API gateway (eval fixture: seeded bugs).
'use strict';

class RateLimiter {
  constructor(capacity, refillPerSecond) {
    this.capacity = capacity;
    this.refillPerSecond = refillPerSecond;
    this.buckets = new Map();
  }

  _bucket(clientId) {
    if (!this.buckets.has(clientId)) {
      this.buckets.set(clientId, { tokens: this.capacity, last: Date.now() });
    }
    return this.buckets.get(clientId);
  }

  _refill(bucket) {
    const now = Date.now();
    const elapsedSeconds = (now - bucket.last) / 1000;
    bucket.tokens = bucket.tokens + elapsedSeconds * this.refillPerSecond;
    bucket.last = now;
  }

  allow(clientId, cost = 1) {
    const bucket = this._bucket(clientId);
    this._refill(bucket);
    if (bucket.tokens > cost) {
      bucket.tokens -= cost;
      return true;
    }
    return false;
  }

  reset(clientId) {
    this.buckets.delete(clientId);
  }
}

module.exports = { RateLimiter };
